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
from dataclasses import dataclass
from pathlib import Path

from . import config_edit as ce
from .tools import Registry
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


def register(r: Registry, ws: Workspace, store: ProposalStore | None = None) -> ProposalStore:
    store = store or ProposalStore()

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
        path, key, inferred = _infer_path(ws, path, key)
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
        if errors or not diff:
            result["proposal_id"] = None  # nothing to apply
            return result
        summary = f"agent: {'delete' if delete else 'set'} {key} in {rel}"
        prop = store.add(path=p, base_sha256=_sha256(p), new_text=new, summary=summary,
                         diff=diff, reason=reason)
        result["proposal_id"] = prop.id
        return result

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
