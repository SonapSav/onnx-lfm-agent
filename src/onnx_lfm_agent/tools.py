from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from jsonschema import Draft202012Validator

MAX_ARG_ERRORS = 5  # keep error feedback short; the 1.2B model drowns in long lists


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments
    func: Callable[..., Any]
    dangerous: bool = False  # side-effecting -> gated by the agent's approve hook
    _validator: Draft202012Validator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Fail fast on a malformed schema at registration, not mid-run.
        Draft202012Validator.check_schema(self.parameters)
        self._validator = Draft202012Validator(self.parameters)

    def validate(self, args: Any) -> list[str]:
        """Check `args` against `parameters`. Returns readable errors; empty if valid."""
        errors = sorted(self._validator.iter_errors(args), key=lambda e: list(e.absolute_path))
        out = []
        for e in errors[:MAX_ARG_ERRORS]:
            path = ".".join(str(p) for p in e.absolute_path)
            out.append(f"{path}: {e.message}" if path else e.message)
        return out


class Registry:
    """Holds the tools the agent may call. Produces OpenAI `tools` schemas and
    dispatches by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, *, description: str, parameters: dict,
             dangerous: bool = False, name: str | None = None):
        """Decorator to register a function as a tool."""
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.add(Tool(name or fn.__name__, description, parameters, fn, dangerous))
            return fn
        return deco

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        return [
            {"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)
