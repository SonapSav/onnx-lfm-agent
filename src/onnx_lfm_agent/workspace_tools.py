"""File and config tools, sandboxed to one workspace directory.

Read-only tools run freely. Config changes follow propose -> validate -> apply:
propose_config_change writes nothing and returns a diff + proposal_id;
apply_config_change (dangerous) writes the file and commits it to git;
rollback is operator-only (rollback_last_change / `lfm-agent --rollback`).
"""

from __future__ import annotations

import hashlib
import itertools
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import config_edit as ce
from . import log_fix
from .client import make_client
from .config import settings
from .tools import CURRENT_REQUEST, CURRENT_RUN, Registry
from .workspace import Workspace, WorkspaceError

MAX_LIST = 200
MAX_LINES = 200
MAX_READ_CHARS = 8000
MAX_MATCHES = 50
MAX_MATCH_LINE = 200
MAX_SEARCH_BYTES = 1_000_000  # skip larger files when searching
MAX_PROPOSALS = 50
TRAILER = "Agent-Proposal"


@dataclass
class Proposal:
    id: str
    path: Path
    base_sha256: str
    new_text: str
    summary: str  # becomes the commit subject
    diff: str
    reason: str
    run: str | None = None  # the Agent.run() that proposed it (tools.CURRENT_RUN)


class ProposalStore:
    """Valid, not-yet-applied proposals, in memory (lost on restart)."""

    def __init__(self, cap: int = MAX_PROPOSALS) -> None:
        self._items: OrderedDict[str, Proposal] = OrderedDict()
        self._ids = itertools.count(1)
        self._cap = cap
        self._lock = threading.Lock()

    def add(self, **fields) -> Proposal:
        with self._lock:
            p = Proposal(id=f"p{next(self._ids)}", **fields)
            self._items[p.id] = p
            while len(self._items) > self._cap:
                self._items.popitem(last=False)
            return p

    def pending_in(self, run: str) -> list[Proposal]:
        """Unapplied proposals made earlier in agent run `run`."""
        with self._lock:
            return [p for p in self._items.values() if p.run == run]

    def peek(self, proposal_id: str) -> Proposal | None:
        with self._lock:
            return self._items.get(proposal_id)

    def pop(self, proposal_id: str) -> Proposal:
        with self._lock:
            try:
                return self._items.pop(proposal_id)
            except KeyError:
                raise WorkspaceError(f"unknown or already used proposal_id: {proposal_id}; "
                                     "call propose_config_change first") from None


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _is_binary(p: Path) -> bool:
    with p.open("rb") as f:
        return b"\0" in f.read(1024)


def _walk(ws: Workspace, base: Path):
    """Files and dirs under `base`, sorted, skipping .git and links leaving the root."""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d != ".git")
        d = Path(dirpath)
        for name in sorted(dirnames):
            yield d / name, True
        for name in sorted(filenames):
            p = d / name
            if p.resolve().is_relative_to(ws.root):
                yield p, False


def _config_files(ws: Workspace) -> list[str]:
    """Editable config files (schemas excluded), workspace-relative."""
    return [ws.rel(p) for p, is_dir in _walk(ws, ws.root)
            if not is_dir and p.suffix.lower() in ce.SUFFIXES
            and not p.name.endswith(".schema.json")]


def _only(files: list[str], kind: str) -> str:
    """Narrow repair for an omitted path: the only candidate, else refuse with the list."""
    if len(files) == 1:
        return files[0]
    raise WorkspaceError(f"which {kind}? candidates: {', '.join(files) or 'none found'}")


def _infer_path(ws: Workspace, path: str, key: str) -> tuple[str, str, str | None]:
    """Narrow repair: the 1.2B model often omits `path` or folds the file name
    into the key ("app.yaml.logging.level"). Returns (path, key, note)."""
    if path:
        rel = ws.rel(ws.resolve(path))
        if key.startswith(rel + "."):
            return path, key[len(rel) + 1:], f"removed the file name from key {key!r}"
        return path, key, None
    configs = _config_files(ws)
    for c in sorted(configs, key=len, reverse=True):  # longest first: a/app.yaml before app.yaml
        if key.startswith(c + "."):
            return c, key[len(c) + 1:], f"path not given; took {c} from key {key!r}"
    if len(configs) == 1:
        return configs[0], key, f"path not given; used the only config file, {configs[0]}"
    raise WorkspaceError("path is required: which config file? "
                         f"candidates: {', '.join(configs) or 'none found'}")


# No system-prompt guidance from this toolset, on purpose. Live evals
# (LFM2.5-1.2B, 6 runs each): workspace instructions never got the model to
# read the config before proposing (open-ended 0/6), and made it apply changes
# it was only asked to propose (0/6, from 6/6 without). A real key as the
# example got copied verbatim. A read-before-propose guard was tried too and
# removed: the extra round derailed it (directed propose 6/6 -> 0-1/6) with no
# open-ended gain. See CLAUDE.md.
GUIDANCE: list[str] = []


def register(r: Registry, ws: Workspace, store: ProposalStore | None = None,
             client=None) -> ProposalStore:
    """`client` makes propose_fix_from_logs' own model calls (default: a client
    for LFM_URL, created on first use)."""
    store = store or ProposalStore()
    r.guidance.extend(GUIDANCE)

    def model() -> tuple:
        """(client, model, temperature) for the harness's own narrow model calls."""
        nonlocal client
        client = client or make_client()
        return client, settings.model, settings.temperature

    def _refuse_if_pending(same: Callable[[Proposal], bool] | None = None) -> dict | None:
        """One open proposal per agent run. Live runs: after a good proposal the
        model proposed again (via either tool, often garbage, sometimes for
        another file) and applied the newer one. A tool error is what it
        follows. Applying the pending one frees it (two changes = propose,
        apply, propose, apply)."""
        run = CURRENT_RUN.get()
        prior = store.pending_in(run) if run else []
        if prior:
            p = prior[-1]
            if same and same(p):  # the identical change again: not an error (A stalled on it)
                return {"path": ws.rel(p.path), "diff": p.diff, "valid": True, "proposal_id": p.id,
                        "note": f"already proposed as {p.id} in this conversation"}
            raise WorkspaceError(
                f"{p.id} is already proposed in this conversation ({p.summary.removeprefix('agent: ')}"
                f"). Do not propose again: if the user asked to apply it, call apply_config_change "
                f"with proposal_id {p.id}; otherwise report it.")

    def _locate(path: str, key: str, value, reason: str) -> tuple[str, str, str | None]:
        """(path, key, note). An invented file name counts as omitted; when the
        file is still ambiguous, the model picks the real setting from all of them
        (with several config files it never supplied a path, even when told the
        candidates)."""
        bad = None
        if path and not ws.resolve(path).is_file():
            bad, path = f"{path} does not exist", ""
        try:
            path, key, note = _infer_path(ws, path, key)
        except WorkspaceError:
            configs = _config_files(ws)
            if len(configs) < 2:
                raise
            s = log_fix.resolve_setting(ws, *model(), configs, CURRENT_REQUEST.get(), key, value, reason)
            if s is None:
                raise WorkspaceError(f"no setting matches {key!r} in {', '.join(configs)}; "
                                     "tell the user") from None
            path, key, note = s.file, s.key, f"{key!r} is not a setting; matched it to {s.key} in {s.file}"
        return path, key, "; ".join(filter(None, (bad, note))) or None

    def _preview(args: dict) -> str:
        prop = store.peek(args.get("proposal_id", ""))
        return f"{prop.summary}\n{prop.diff}" if prop else "(unknown proposal_id)"

    @r.tool(
        description="List files and folders in the workspace (recursive). Folders end with '/'.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "Folder to list, relative to the workspace. Default '.'"},
        }},
    )
    def list_files(path: str = ".") -> dict:
        base = ws.resolve(path)
        if not base.is_dir():
            raise WorkspaceError(f"not a folder: {path}")
        entries = []
        for p, is_dir in _walk(ws, base):
            if len(entries) == MAX_LIST:
                return {"entries": entries, "truncated": True}
            entries.append(ws.rel(p) + ("/" if is_dir else ""))
        return {"entries": entries, "truncated": False}

    @r.tool(
        description="Read a text file from the workspace. Returns numbered lines.",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace"},
            "start_line": {"type": "integer", "minimum": 1, "description": "First line to read. Default 1"},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": MAX_LINES,
                          "description": f"How many lines. Default {MAX_LINES}"},
        }, "required": ["path"]},
    )
    def read_file(path: str, start_line: int = 1, max_lines: int = MAX_LINES) -> dict:
        p = ws.resolve(path)
        if not p.is_file():
            raise WorkspaceError(f"not a file: {path}")
        if _is_binary(p):
            raise WorkspaceError(f"binary file, not readable as text: {path}")
        lines = p.read_text(errors="replace").splitlines()
        out, chars, end = [], 0, start_line - 1
        for n in range(start_line - 1, min(len(lines), start_line - 1 + max_lines)):
            line = f"{n + 1}: {lines[n]}"
            if chars + len(line) > MAX_READ_CHARS:
                break
            out.append(line)
            chars += len(line) + 1
            end = n + 1
        return {"path": ws.rel(p), "content": "\n".join(out), "total_lines": len(lines),
                "truncated": end < len(lines)}

    @r.tool(
        description="Find lines containing some text (case-insensitive) in workspace files.",
        parameters={"type": "object", "properties": {
            "text": {"type": "string", "minLength": 1, "description": "Text to look for"},
            "path": {"type": "string", "description": "File or folder to search. Default '.'"},
        }, "required": ["text"]},
    )
    def search_files(text: str, path: str = ".") -> dict:
        base = ws.resolve(path)
        files = [base] if base.is_file() else [p for p, is_dir in _walk(ws, base) if not is_dir]
        needle, matches = text.lower(), []
        for f in files:
            if f.stat().st_size > MAX_SEARCH_BYTES or _is_binary(f):
                continue
            for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if needle in line.lower():
                    if len(matches) == MAX_MATCHES:
                        return {"matches": matches, "truncated": True}
                    matches.append(f"{ws.rel(f)}:{n}: {line.strip()[:MAX_MATCH_LINE]}")
        return {"matches": matches, "truncated": False}

    @r.tool(
        description=("Propose changing ONE key in a JSON/YAML config file. Changes nothing: "
                     "returns a diff, validation result and a proposal_id to pass to "
                     "apply_config_change. Key is dotted, e.g. server.port"),
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "Config file path relative to the workspace"},
            "key": {"type": "string", "minLength": 1, "description": "Dotted key, e.g. server.port or workers.0.name"},
            "value": {"description": "New value, typed like the existing one: numbers and "
                                     "true/false unquoted (15, not \"15\")"},
            "delete": {"type": "boolean", "description": "true to remove the key instead of setting it"},
            "reason": {"type": "string", "description": "Short reason, e.g. what in the logs motivated it"},
        }, "required": ["key"]},  # path is inferred when omitted (see _infer_path)
    )
    def propose_config_change(key: str, path: str = "", value=ce.DELETE, delete: bool = False,
                              reason: str = "") -> dict:
        path, key, inferred = _locate(path, key, value, reason)
        result = _propose(path, key, value, delete, reason, inferred, check_pending=True)
        if result.pop("unknown_key", False):
            # The key doesn't exist and the schema won't take it as a new one:
            # let the model pick the setting it meant in that file (seen:
            # "services.worker.concurrency" for queues.1.concurrency).
            s = log_fix.resolve_setting(ws, *model(), [result["path"]], CURRENT_REQUEST.get(),
                                        key, value, reason)
            if s is not None:
                note = f"{key!r} is not a setting; matched it to {s.key}"
                return _propose(s.file, s.key, value, delete, reason,
                                "; ".join(filter(None, (inferred, note))), check_pending=True)
        return result

    def _propose(path: str, key: str, value, delete: bool, reason: str,
                 inferred: str | None, check_pending: bool = False) -> dict:
        p = ws.resolve(path)
        fmt = ce.config_format(p)
        if not p.is_file():
            raise WorkspaceError(f"not a file: {path}")
        if delete == (value is not ce.DELETE):
            raise WorkspaceError("give either a value or delete=true (not both, not neither)")
        ws.require_git()
        rel = ws.rel(p)
        if ws.git("status", "--porcelain", "--", rel).stdout.strip():
            raise WorkspaceError(f"{rel} has uncommitted changes; commit or discard them first")

        old = p.read_text()

        def attempt(v):
            data = ce.edit(ce.parse(old, fmt), key, v)
            return (data, *ce.validate(data, p))

        data, errors, schema = attempt(value)
        note = None
        # Narrow repair: the 1.2B model often quotes numbers/booleans ("15").
        # Convert only if the schema rejects the string and accepts the
        # converted value, or (no schema) the current value is that kind.
        alt = ce.scalar_from_string(value)
        if alt is not ce.MISSING:
            if schema:
                convert = bool(errors) and not attempt(alt)[1]
            else:
                current = ce.lookup(ce.parse(old, fmt), key)
                convert = current is not ce.MISSING and ce.same_kind(alt, current)
            if convert:
                data, errors, _ = attempt(alt)
                why = f"to match {schema}" if schema else "to match the current value's type"
                note = f"converted {value!r} (string) to {alt!r} {why}"
        new = ce.dump(data, fmt, old)
        diff = ce.diff(old, new, rel)
        if check_pending and (repeat := _refuse_if_pending(lambda q: q.path == p and q.new_text == new)):
            return repeat
        note = "; ".join(filter(None, (inferred, note))) or None
        result = {"path": rel, "diff": diff or "(no change)", "valid": not errors,
                  "errors": errors, "schema": schema or "none found (only checked that it parses)"}
        if note:
            result["note"] = note
        if errors and value is not ce.DELETE:
            parent = key.rpartition(".")[0]
            original = ce.parse(old, fmt)
            if ce.lookup(original, key) is ce.MISSING:  # a new key the schema may not allow
                node = ce.lookup(original, parent) if parent else original
                result["hint"] = f"{key} is a new key; {ce.existing_keys(node, parent)}"
                result["unknown_key"] = True
        if errors or not diff:
            result["proposal_id"] = None  # nothing to apply
            return result
        summary = f"agent: {'delete' if delete else 'set'} {key} in {rel}"
        prop = store.add(path=p, base_sha256=_sha256(p), new_text=new, summary=summary,
                         diff=diff, reason=reason, run=CURRENT_RUN.get())
        result["proposal_id"] = prop.id
        return result

    @r.tool(
        description=("Find the errors in a log file and propose ONE config change that fixes "
                     "them. Changes nothing: returns a diff and a proposal_id to pass to "
                     "apply_config_change."),
        parameters={"type": "object", "properties": {
            "log_path": {"type": "string", "description": "Log file, e.g. logs/app.log"},
            "config_path": {"type": "string", "description": "Config file to change"},
        }},  # both inferred when omitted: the only .log / config file
    )
    def propose_fix_from_logs(log_path: str = "", config_path: str = "") -> dict:
        request = CURRENT_REQUEST.get()
        # Invented paths (seen: config.yaml, logs/app.log) count as omitted.
        if not (log_path and ws.resolve(log_path).is_file()):
            logs = [ws.rel(p) for p, is_dir in _walk(ws, ws.root)
                    if not is_dir and p.suffix.lower() == ".log"]
            log_path = log_fix.choose_log(ws, *model(), logs, request)
        if config_path and ws.resolve(config_path).is_file():
            configs = [ws.rel(ws.resolve(config_path))]
        else:
            configs = _config_files(ws)  # several: the key choice also picks the file
        # Live runs: asked "set X to 15, then apply", the model sometimes also
        # called this tool (the request mentions logs), got a second proposal
        # (10) and applied that one. A tool error is what it follows.
        _refuse_if_pending()
        return log_fix.run(ws, *model(), propose_config_change, log_path, configs, request)

    @r.tool(
        description=("Apply a proposal from propose_config_change: writes the file and "
                     "commits it to git. Needs the proposal_id."),
        parameters={"type": "object", "properties": {
            "proposal_id": {"type": "string", "description": "e.g. p1"},
        }, "required": ["proposal_id"]},
        dangerous=True,
        preview=_preview,
    )
    def apply_config_change(proposal_id: str) -> dict:
        with ws.lock:
            run = CURRENT_RUN.get()
            if store.peek(proposal_id) is None and run:
                # Seen: invented ids ("p123"). Name the real ones from this run.
                mine = [p.id for p in store.pending_in(run)]
                raise WorkspaceError(f"unknown or already used proposal_id: {proposal_id}; "
                                     f"pending in this conversation: {', '.join(mine) or 'none'}")
            prop = store.pop(proposal_id)
            rel = ws.rel(prop.path)
            if not prop.path.is_file() or _sha256(prop.path) != prop.base_sha256:
                raise WorkspaceError(f"{rel} changed since {proposal_id} was proposed; propose again")
            original = prop.path.read_bytes()
            prop.path.write_text(prop.new_text)
            body = f"{prop.reason}\n\n{TRAILER}: {prop.id}" if prop.reason else f"{TRAILER}: {prop.id}"
            try:
                ws.git("add", "--", rel)
                ws.git("commit", "-q", "-m", prop.summary, "-m", body, "--", rel)
            except WorkspaceError:
                prop.path.write_bytes(original)  # leave the tree as we found it
                ws.git("reset", "-q", "--", rel, check=False)
                raise
            commit = ws.git("rev-parse", "--short", "HEAD").stdout.strip()
        return {"applied": proposal_id, "path": rel, "commit": commit}

    return store


def last_change(ws: Workspace) -> dict:
    """The latest commit, and whether apply_config_change made it."""
    ws.require_git()
    short, subject, body = ws.git("log", "-1", "--format=%h%x00%s%x00%B").stdout.split("\0", 2)
    return {"commit": short, "subject": subject,
            "by_agent": f"\n{TRAILER}: " in f"\n{body}"}


def rollback_last_change(ws: Workspace) -> dict:
    """Revert the latest commit if apply_config_change made it.

    Operator-only (``lfm-agent --rollback``), deliberately not a model tool: in
    live runs the 1.2B model called rollback when asked to apply.
    """
    with ws.lock:
        head = last_change(ws)
        if not head["by_agent"]:
            raise WorkspaceError(f"latest commit {head['commit']} ({head['subject']!r}) was not "
                                 "made by apply_config_change; refusing to revert it")
        ws.git("revert", "--no-edit", "HEAD")
        commit = ws.git("rev-parse", "--short", "HEAD").stdout.strip()
    return {"reverted": head["commit"], "subject": head["subject"], "commit": commit}
