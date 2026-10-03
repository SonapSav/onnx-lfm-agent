"""Demo tools (toolset "demo"): safe, read-only, useful for smoke tests."""

from __future__ import annotations

import datetime

from .tools import Registry


def register(r: Registry) -> None:
    @r.tool(
        description="Return the current local date and time.",
        parameters={"type": "object", "properties": {}},
    )
    def get_current_time() -> dict:
        return {"now": datetime.datetime.now().isoformat(timespec="seconds")}

    @r.tool(
        description="Add two numbers.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )
    def add(a: float, b: float) -> dict:
        return {"sum": a + b}
