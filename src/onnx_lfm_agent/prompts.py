"""The system prompt: a short base plus guidance contributed by each toolset.

Every token here is re-prefilled on every model call (~6 ms/token on this
host), and a 1.2B model follows short imperative lines best — keep it tight.
"""

from __future__ import annotations

from .tools import Registry

BASE = [
    "You are a concise assistant with tools. Use a tool only when the request needs one.",
    "If a tool returns an error, fix the arguments and try again.",
]


def build_system_prompt(registry: Registry) -> str:
    return "\n".join(f"- {line}" for line in [*BASE, *registry.guidance])
