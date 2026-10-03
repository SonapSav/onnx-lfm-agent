"""Harness-driven workflow: log errors -> one config change, as a proposal.

LFM2.5-1.2B can't plan "read the logs and fix the config" on its own (0/6 in
live evals under every prompt variant: it guesses key names and never uses
what it reads). Here code does the planning and the model fills two narrow
decisions, each a single forced-shape tool call:

  1. code: error/warning lines from the log, deduplicated
  2. code: the config's leaf keys with values, schema limits and comments
  3. model: which key (an enum of the real keys, so no invented names)
  4. model: the new value (typed from the schema; code rejects "unchanged"
     and out-of-range values and lets it retry, since it follows tool errors)
  5. code: propose_config_change (diff + validation + proposal_id; no write)

Probing showed why 3 and 4 are separate: asked for key and value together it
picked the right key 8/8 but copied the current value back 8/8; asked for the
value alone it raised it 6/6.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from jsonschema import Draft202012Validator

from . import config_edit as ce
from .workspace import Workspace, WorkspaceError

MAX_EVIDENCE = 10  # unique log lines shown to the model
MAX_LINE = 200
MAX_TRIES = 3  # per decision
SIGNAL = re.compile(r"\b(ERROR|ERR|WARN|WARNING|FATAL|CRITICAL|PANIC)\b|exception|traceback",
                    re.IGNORECASE)
# Leading "2026-10-02 14:05:40" / ISO timestamps, so repeats of one error collapse.
TIMESTAMP = re.compile(r"^\[?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\]?\s*")


def evidence(text: str) -> list[str]:
    """Error/warning lines, timestamps stripped, deduplicated with counts,
    in first-seen order: ["3x ERROR upstream ... timed out after 5s", ...]."""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        if not SIGNAL.search(line):
            continue
        msg = TIMESTAMP.sub("", line.strip())[:MAX_LINE]
        counts[msg] = counts.get(msg, 0) + 1
    return [f"{n}x {msg}" if n > 1 else msg for msg, n in list(counts.items())[:MAX_EVIDENCE]]


def leaves(data: Any, prefix: str = "") -> dict[str, Any]:
    """{dotted key: scalar value} for every leaf; lists are indexed."""
    if isinstance(data, dict):
        items = ((str(k), v) for k, v in data.items())
    elif isinstance(data, list):
        items = ((str(i), v) for i, v in enumerate(data))
    else:
        return {prefix: data}
    out: dict[str, Any] = {}
    for k, v in items:
        out.update(leaves(v, f"{prefix}.{k}" if prefix else k))
    return out


def _comment(data: Any, key: str) -> str:
    """The end-of-line YAML comment on `key`, if any (ruamel round-trip data)."""
    *parents, last = key.split(".")
    node = data
    for part in parents:
        node = node[int(part)] if isinstance(node, list) else node[part]
    ca = getattr(node, "ca", None)
    entry = ca.items.get(int(last) if isinstance(node, list) else last) if ca else None
    token = next((t for t in (entry or [])[2:3] if t is not None), None)
    text = token.value.strip() if token else ""  # may be just the blank lines after a value
    return text.lstrip("#").strip().splitlines()[0] if text.startswith("#") else ""


def _subschema(schema: dict, key: str) -> dict:
    node = schema
    for part in key.split("."):
        if part.isdigit() and "items" in node:
            node = node["items"]
        else:
            node = node.get("properties", {}).get(part)
        if not isinstance(node, dict):
            return {}
    return node


def _describe(sub: dict) -> str:
    """'number, 1..120' / 'one of debug, info' — the schema limits in words."""
    if "enum" in sub:
        return "one of " + ", ".join(map(str, sub["enum"]))
    bits = [sub["type"]] if isinstance(sub.get("type"), str) else []
    lo, hi = sub.get("minimum"), sub.get("maximum")
    if lo is not None or hi is not None:
        bits.append(f"{'' if lo is None else lo}..{'' if hi is None else hi}")
    return ", ".join(bits)


def _value_schema(sub: dict) -> dict:
    """The value parameter: the key's own schema when it is a simple one
    (single type or enum), else untyped. A multi-type list is what made the
    model send 0 for "debug" in propose_config_change."""
    if "enum" in sub or isinstance(sub.get("type"), str):
        return {k: v for k, v in sub.items() if k in ("type", "enum", "minimum", "maximum")}
    return {}


def _decide(client, model: str, temperature: float, prompt: str, tool: dict,
            check: Callable[[dict], str | None]) -> dict:
    """One narrow decision: a single tool call, retried with the error fed back
    (the model follows tool errors far better than instructions)."""
    messages: list[dict] = [{"role": "user", "content": prompt}]
    params = tool["function"]["parameters"]
    validator = Draft202012Validator(params)
    for _ in range(MAX_TRIES):
        msg = client.chat.completions.create(model=model, messages=messages, tools=[tool],
                                             temperature=temperature).choices[0].message
        if not msg.tool_calls:
            messages += [{"role": "assistant", "content": msg.content or ""},
                         {"role": "user", "content": f"Answer by calling {tool['function']['name']}."}]
            continue
        tc = msg.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args, error = None, "could not parse the arguments as JSON"
        else:
            errors = [e.message for e in validator.iter_errors(args)]
            error = "; ".join(errors[:3]) if errors else check(args)
        if error is None:
            return args
        messages += [{"role": "assistant", "content": None, "tool_calls": [
                         {"id": tc.id, "type": "function",
                          "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]},
                     {"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"error": error})}]
    raise WorkspaceError(f"the model did not make a valid {tool['function']['name']} choice "
                         f"in {MAX_TRIES} tries")


NONE = "none"  # key-choice option: no listed setting fixes it / matches the request


@dataclass
class Setting:
    file: str  # workspace-relative config file
    key: str  # dotted key inside it
    value: Any
    schema: dict  # the key's own sub-schema ({} if none)
    notes: str  # "number, 1..120; <YAML comment>"


def settings_index(ws: Workspace, files: list[str]) -> dict[str, Setting]:
    """{label: Setting} for every scalar leaf of `files`. Labels are the bare
    key for one file, "file: key" for several (so the key choice also picks
    the file — the model never supplied a path when it had to)."""
    out: dict[str, Setting] = {}
    for rel in files:
        path = ws.resolve(rel)
        data = ce.parse(path.read_text(), ce.config_format(path))
        sp = ce.schema_path(path)
        schema = json.loads(sp.read_text()) if sp.is_file() else {}
        for key, value in leaves(data).items():
            if value is not None and not isinstance(value, (str, int, float, bool)):
                continue
            sub = _subschema(schema, key)
            notes = "; ".join(filter(None, (_describe(sub), _comment(data, key))))
            named = _named_path(data, key)
            label = named if len(files) == 1 else f"{rel}: {named}"
            out[label] = Setting(rel, key, value, sub, notes)
    return out


def _named_path(data: Any, key: str) -> str:
    """queues.1.concurrency -> queues.reports.concurrency when queues.1 has
    name "reports". Asked for "the reports queue", the model picked queues.0
    from bare indices, and mangled labels like "queues.1.concurrency [reports]"."""
    parts, node, out = key.split("."), data, []
    for i, part in enumerate(parts):
        node = node[int(part)] if isinstance(node, list) else node[part]
        name = (next((node[k] for k in ("name", "id", "title") if isinstance(node.get(k), str)), None)
                if part.isdigit() and isinstance(node, dict) else None)
        out.append(name if name and "." not in name else part)
    return ".".join(out)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


def mentioned_first(index: dict[str, Setting], lines: list[str]) -> dict[str, Setting]:
    """Settings the error lines name go first, marked "named in the log"; then
    ones sharing a word with them ("related: pool"). With several files the
    model otherwise picked a prominent key from the first file
    (database.pool_size) over the one the error names (max_payload_mb=8)."""
    text = "\n".join(lines)
    seen = _words(text)
    named, related = {}, {}
    for label, s in index.items():
        leaf = s.key.rsplit(".", 1)[-1]
        if re.search(rf"\b{re.escape(leaf)}\b", text, re.IGNORECASE):  # "port" is not in "reports"
            named[label] = "named in the log"
        elif shared := sorted(_words(leaf.replace("_", " ")) & seen):
            related[label] = "related: " + ", ".join(shared)
    def mark(label: str, why: str) -> Setting:
        s = index[label]
        return Setting(s.file, s.key, s.value, s.schema, "; ".join(filter(None, (why, s.notes))))
    return {**{k: mark(k, w) for k, w in named.items()}, **{k: mark(k, w) for k, w in related.items()},
            **{k: s for k, s in index.items() if k not in named and k not in related}}


def _reported_above(lines: list[str], current: Any, schema: dict) -> float | None:
    """The largest number the log lines naming a numeric setting report above
    its current value, within the schema's bounds ("payload 12.4 MB exceeds
    max_payload_mb=8" -> 12.4; job ids like 1832 fall outside 1..100). The
    model can't infer it: shown that line, it chose 9 or 10 under every phrasing.
    Only for bounded numeric settings, so a stray id can't become a floor."""
    if (isinstance(current, bool) or not isinstance(current, (int, float))
            or "maximum" not in schema):
        return None
    lo, hi = schema.get("minimum", float("-inf")), schema["maximum"]
    seen = [float(n) for line in lines for n in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", line)]
    above = [n for n in seen if current < n and lo <= n <= hi]
    return max(above) if above else None


def _show(v: Any) -> str:
    return json.dumps(v, default=str)


def _listing(index: dict[str, Setting]) -> str:
    return "\n".join(f"{label} = {_show(s.value)}" + (f"  # {s.notes}" if s.notes else "")
                     for label, s in index.items())


def _asked(request: str | None) -> str:
    return f"User request: {request}\n\n" if request else ""


def _choose_setting(client, model, temperature, prompt: str, index: dict[str, Setting]) -> Setting | None:
    label = _decide(client, model, temperature, prompt,
                    {"type": "function", "function": {
                        "name": "choose_setting", "description": "Choose the setting.",
                        "parameters": {"type": "object", "properties": {
                            "key": {"type": "string", "enum": [*index, NONE]},
                            "reason": {"type": "string"}}, "required": ["key"]}}},
                    lambda a: None)["key"]
    return None if label == NONE else index[label]


def choose_log(ws: Workspace, client, model: str, temperature: float, logs: list[str],
               request: str | None) -> str:
    """The log to read when the model gave none (or one that doesn't exist)
    and there are several: a pick from the real files, shown with their errors."""
    if len(logs) == 1:
        return logs[0]
    if not logs:
        raise WorkspaceError("no .log files in the workspace")
    # Code decides when exactly one log shares the request's distinguishing
    # words: asked "the API keeps returning 503s", the model picked worker.log
    # 6/6 under every phrasing, even with api.log first and marked as matching.
    found = {rel: evidence(ws.resolve(rel).read_text(errors="replace")) for rel in logs}
    vocab = {rel: _words(rel + " " + " ".join(lines)) for rel, lines in found.items()}
    common = set.intersection(*vocab.values())  # e.g. "logs", "error": no signal
    asked, ranked = _words(request or "") - common, []
    for rel, lines in found.items():
        shared = sorted(asked & vocab[rel])
        ranked.append((-len(shared), rel, shared, lines))
    matching = [rel for n, rel, _, _ in ranked if n]
    if len(matching) == 1:
        return matching[0]
    shown = []
    for _, rel, shared, lines in sorted(ranked):
        head = f"{rel}" + (f" (matches the request: {', '.join(shared)})" if shared else "")
        shown.append(f"{head}:\n" + ("\n".join(f"  {x}" for x in lines[:4]) or "  (no errors)"))
    logs = [rel for _, rel, _, _ in sorted(ranked)]
    return _decide(client, model, temperature,
                   _asked(request) + "Log files and their errors:\n" + "\n".join(shown)
                   + "\n\nWhich log file is about the user's problem?",
                   {"type": "function", "function": {
                       "name": "choose_log", "description": "Choose the log file.",
                       "parameters": {"type": "object", "properties": {
                           "path": {"type": "string", "enum": logs}}, "required": ["path"]}}},
                   lambda a: None)["path"]


def resolve_setting(ws: Workspace, client, model: str, temperature: float, files: list[str],
                    request: str | None, key: str, value: Any, reason: str) -> Setting | None:
    """propose_config_change got a key that doesn't exist, or no usable file
    among several: let the model pick the real setting it meant (or none)."""
    index = settings_index(ws, files)
    tried = f"An assistant tried to change {key!r} to {_show(value)}" + (f" ({reason})" if reason else "")
    return _choose_setting(client, model, temperature,
                           _asked(request) + f"{tried}, but no setting has that name.\n\n"
                           f"Settings:\n{_listing(index)}\n\n"
                           f"Which setting did the user mean? Choose {NONE!r} if none of them.",
                           index)


def run(ws: Workspace, client, model: str, temperature: float, propose: Callable[..., dict],
        log_path: str, configs: list[str], request: str | None = None) -> dict:
    log = ws.resolve(log_path)
    if not log.is_file():
        raise WorkspaceError(f"not a file: {log_path}")
    lines = evidence(log.read_text(errors="replace"))
    if not lines:
        return {"log": ws.rel(log), "evidence": [], "proposal_id": None,
                "result": "no error or warning lines found; nothing to fix"}
    index = settings_index(ws, configs)
    if not index:
        raise WorkspaceError(f"no settings to change in {', '.join(configs)}")
    if len(configs) > 1:
        index = mentioned_first(index, lines)

    log_block = f"Log lines from {ws.rel(log)}:\n" + "\n".join(lines)
    where = configs[0] if len(configs) == 1 else "the config files"
    setting = _choose_setting(
        client, model, temperature,
        _asked(request) + f"{log_block}\n\nSettings in {where}:\n{_listing(index)}\n\n"
        f"Which one setting should change to fix the errors? Choose {NONE!r} if changing a "
        "setting would not fix them.", index)
    if setting is None:
        # Seen: auth errors (wrong passwords) -> the model changed server.port.
        return {"log": ws.rel(log), "evidence": lines, "proposal_id": None,
                "result": "no config change would fix these errors",
                "next": "Report this to the user. Do not propose or apply a config change."}

    old, key, rel = setting.value, setting.key, setting.file
    value_param = {**_value_schema(setting.schema), "description": "New value"}

    leaf = key.rsplit(".", 1)[-1]
    naming = [x for x in lines if re.search(rf"\b{re.escape(leaf)}\b", x, re.IGNORECASE)]
    floor = _reported_above(naming, old, setting.schema)

    def changed(a: dict) -> str | None:
        v = ce.scalar_from_string(a["value"])
        v = a["value"] if v is ce.MISSING else v
        if v == old:
            return f"{key} is already {old!r}; choose a different value"
        if floor is not None and isinstance(v, (int, float)) and v < floor:
            return f"the log reports {floor:g} for {key}; choose at least {floor:g}"
        return None

    notes = f"  # {setting.notes}" if setting.notes else ""
    focus = ("\nThe log lines about this setting:\n" + "\n".join(naming) + "\n") if naming else ""
    value = _decide(client, model, temperature,
                    _asked(request) + f"{log_block}\n\nThe setting {key} in {rel} is currently "
                    f"{_show(old)}{notes}\n{focus}It must change so that these errors stop. "
                    "Choose the new value.",
                    {"type": "function", "function": {
                        "name": "set_value", "description": "Set the new value of the setting.",
                        "parameters": {"type": "object", "properties": {"value": value_param},
                                       "required": ["value"]}}},
                    changed)["value"]

    reason = lines[0] + (f" (+{len(lines) - 1} more log lines)" if len(lines) > 1 else "")
    result = propose(path=rel, key=key, value=value, reason=reason)
    out = {"log": ws.rel(log), "evidence": lines, **result}
    if result.get("proposal_id"):
        # Live runs: without this the outer model re-proposed on its own
        # (garbage like server.port=10) and applied that instead.
        out["summary"] = f"proposed {key}: {_show(old)} -> {_show(value)} in {rel} (not applied)"
        out["next"] = ("Done. Report this proposal to the user. Only if they asked to apply it, "
                       f"call apply_config_change with proposal_id {result['proposal_id']}.")
    return out
