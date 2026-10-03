from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments
    func: Callable[..., Any]
    dangerous: bool = False  # side-effecting -> gated by the agent's approve hook


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
