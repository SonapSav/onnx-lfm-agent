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


def run(ws: Workspace, client, model: str, temperature: float, propose: Callable[..., dict],
        log_path: str, config_path: str) -> dict:
    log = ws.resolve(log_path)
    if not log.is_file():
        raise WorkspaceError(f"not a file: {log_path}")
    lines = evidence(log.read_text(errors="replace"))
    if not lines:
        return {"log": ws.rel(log), "evidence": [], "proposal_id": None,
                "result": "no error or warning lines found; nothing to fix"}
    cfg = ws.resolve(config_path)
    if not cfg.is_file():
        raise WorkspaceError(f"not a file: {config_path}")
    data = ce.parse(cfg.read_text(), ce.config_format(cfg))
    current = {k: v for k, v in leaves(data).items() if v is None or isinstance(v, (str, int, float, bool))}
    if not current:
        raise WorkspaceError(f"{ws.rel(cfg)} has no settings to change")
    sp = ce.schema_path(cfg)
    schema = json.loads(sp.read_text()) if sp.is_file() else {}
    rel = ws.rel(cfg)

    def notes(k: str) -> str:
        text = "; ".join(filter(None, (_describe(_subschema(schema, k)), _comment(data, k))))
        return f"  # {text}" if text else ""

    def show(v: Any) -> str:
        return json.dumps(v, default=str)

    log_block = f"Log lines from {ws.rel(log)}:\n" + "\n".join(lines)
    key = _decide(client, model, temperature,
                  f"{log_block}\n\nSettings in {rel}:\n"
                  + "\n".join(f"{k} = {show(v)}{notes(k)}" for k, v in current.items())
                  + "\n\nWhich one setting should change to fix the errors?",
                  {"type": "function", "function": {
                      "name": "choose_setting", "description": "Choose the setting to change.",
                      "parameters": {"type": "object", "properties": {
                          "key": {"type": "string", "enum": list(current)},
                          "reason": {"type": "string"}}, "required": ["key"]}}},
                  lambda a: None)["key"]

    old = current[key]
    sub = _subschema(schema, key)
    value_param = {**_value_schema(sub), "description": "New value"}

    def changed(a: dict) -> str | None:
        v = ce.scalar_from_string(a["value"])
        v = a["value"] if v is ce.MISSING else v
        return f"{key} is already {old!r}; choose a different value" if v == old else None

    value = _decide(client, model, temperature,
                    f"{log_block}\n\nThe setting {key} in {rel} is currently {show(old)}{notes(key)}\n"
                    "It must change to fix the errors. Choose the new value.",
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
        out["summary"] = f"proposed {key}: {show(old)} -> {show(value)} in {rel} (not applied)"
        out["next"] = ("Done. Report this proposal to the user. Only if they asked to apply it, "
                       f"call apply_config_change with proposal_id {result['proposal_id']}.")
    return out

