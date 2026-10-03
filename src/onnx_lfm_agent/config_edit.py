"""Single-key edits of JSON/YAML config text: parse, edit, re-serialize, validate.

Edits are one dotted key at a time (e.g. ``server.port``, ``workers.0.name``)
rather than whole-file rewrites: a 1.2B model emitting a full file as a tool
argument is unreliable and hits the API's max_tokens.
"""

from __future__ import annotations

import difflib
import io
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from ruamel.yaml import YAML
from ruamel.yaml.util import load_yaml_guess_indent

from .tools import MAX_ARG_ERRORS

SUFFIXES = {".json": "json", ".yaml": "yaml", ".yml": "yaml"}
DELETE = object()  # sentinel value: remove the key instead of setting it
MISSING = object()  # lookup() result when the key doesn't exist


class ConfigError(Exception):
    """The edit can't be made (bad key, unparseable file, ...)."""


def config_format(path: Path) -> str:
    try:
        return SUFFIXES[path.suffix.lower()]
    except KeyError:
        raise ConfigError(f"not a config file (need .json, .yaml or .yml): {path.name}") from None


def _yaml(original: str = "") -> YAML:
    y = YAML()  # round-trip: keeps comments, key order and quoting
    y.preserve_quotes = True
    if original:
        y.indent(**_yaml_indent(original))
    return y


def _yaml_indent(text: str) -> dict:
    """Match the file's indentation so a one-key edit is a one-line diff.
    ruamel's guess is driven by block sequences; mappings are measured here."""
    mapping = next((len(line) - len(line.lstrip(" ")) for line in text.splitlines()
                    if line.startswith(" ") and not line.lstrip().startswith(("-", "#"))), 2)
    try:
        _, seq, offset = load_yaml_guess_indent(text)
    except Exception:  # noqa: BLE001 — parse errors are reported by parse()
        seq, offset = None, None
    if offset is None:  # no block sequences in the file
        seq, offset = mapping, 0
    return {"mapping": mapping, "sequence": seq, "offset": offset}


def parse(text: str, fmt: str) -> Any:
    try:
        return json.loads(text) if fmt == "json" else _yaml().load(text)
    except Exception as e:  # noqa: BLE001 — any parser error means "can't edit this"
        raise ConfigError(f"file does not parse as {fmt.upper()}: {e}") from e


def dump(data: Any, fmt: str, original: str) -> str:
    if fmt == "json":
        indent = _json_indent(original)
        return json.dumps(data, indent=indent, ensure_ascii=False) + "\n"
    buf = io.StringIO()
    _yaml(original).dump(data, buf)
    return buf.getvalue()


def _json_indent(text: str) -> int:
    for line in text.splitlines()[1:]:
        stripped = line.lstrip(" ")
        if stripped and len(stripped) < len(line):
            return len(line) - len(stripped)
    return 2


def edit(data: Any, key: str, value: Any) -> Any:
    """Set (or, with value=DELETE, remove) the dotted `key` in `data`, in place.

    Intermediate keys must exist; only the last segment may be created.
    Numeric segments index lists.
    """
    parts = key.split(".") if key else []
    if not parts or any(p == "" for p in parts):
        raise ConfigError(f"bad key {key!r}; use dotted form like server.port")
    node = data
    for i, part in enumerate(parts[:-1]):
        node = _child(node, part, ".".join(parts[: i + 1]))
    last = parts[-1]
    if isinstance(node, list):
        idx = _index(node, last, key)
        if value is DELETE:
            del node[idx]
        else:
            node[idx] = value
    elif isinstance(node, dict):
        if value is DELETE:
            if last not in node:
                raise ConfigError(f"key not found: {key} "
                                  f"({existing_keys(node, key.rpartition('.')[0])})")
            del node[last]
        else:
            node[last] = value
    else:
        raise ConfigError(f"cannot set {key}: parent is not an object or list")
    return data


def existing_keys(node: Any, parent: str) -> str:
    """'keys under server: host, port' — feedback so the model can retry."""
    where = parent or "the top level"
    if isinstance(node, dict):
        return f"keys under {where}: {', '.join(map(str, node)) or '(none)'}"
    if isinstance(node, list):
        return f"{where} is a list with {len(node)} items (use 0..{len(node) - 1})"
    return f"{where} is a plain value"


def _child(node: Any, part: str, so_far: str) -> Any:
    if isinstance(node, dict):
        if part not in node:
            parent = so_far.rpartition(".")[0]
            raise ConfigError(f"key not found: {so_far} ({existing_keys(node, parent)})")
        return node[part]
    if isinstance(node, list):
        return node[_index(node, part, so_far)]
    raise ConfigError(f"key not found: {so_far} (parent is a plain value)")


def _index(node: list, part: str, key: str) -> int:
    if not part.isdigit() or int(part) >= len(node):
        raise ConfigError(f"bad list index in {key!r} (list has {len(node)} items)")
    return int(part)


def lookup(data: Any, key: str) -> Any:
    """Current value at dotted `key`, or MISSING."""
    node = data
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return MISSING
    return node


def scalar_from_string(value: Any) -> Any:
    """"15" -> 15, "1.5" -> 1.5, "true" -> True; MISSING if `value` isn't a
    string spelling of a number or boolean. The small model often quotes these."""
    if not isinstance(value, str):
        return MISSING
    text = value.strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        out = json.loads(text)
    except ValueError:
        return MISSING
    return out if isinstance(out, (int, float)) and not isinstance(out, bool) else MISSING


def same_kind(a: Any, b: Any) -> bool:
    """Both booleans, or both (non-boolean) numbers."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool)
    return isinstance(a, (int, float)) and isinstance(b, (int, float))


def schema_path(path: Path) -> Path:
    """app.yaml / app.json -> app.schema.json next to it."""
    return path.with_name(f"{path.stem}.schema.json")


def validate(data: Any, path: Path) -> tuple[list[str], str | None]:
    """(errors, schema file name or None if there is no schema)."""
    sp = schema_path(path)
    if not sp.is_file():
        return [], None
    try:
        schema = json.loads(sp.read_text())
        Draft202012Validator.check_schema(schema)
    except Exception as e:  # noqa: BLE001 — a broken schema must block, not pass
        return [f"schema {sp.name} is unusable: {e}"], sp.name
    # Round-trip through JSON so ruamel's YAML node types validate as plain types.
    plain = json.loads(json.dumps(data, default=str))
    errors = sorted(Draft202012Validator(schema).iter_errors(plain),
                    key=lambda e: list(e.absolute_path))
    out = []
    for e in errors[:MAX_ARG_ERRORS]:
        where = ".".join(str(p) for p in e.absolute_path)
        out.append(f"{where}: {e.message}" if where else e.message)
    return out, sp.name


def diff(old: str, new: str, name: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{name}", tofile=f"b/{name}"))
