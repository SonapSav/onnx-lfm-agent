from __future__ import annotations

import json
from typing import Callable

from .client import make_client
from .config import settings
from .tools import Registry, Tool

# Return True to allow a (dangerous) tool call. Called only for tools marked
# dangerous=True; read-only tools run without prompting.
ApproveFn = Callable[[Tool, dict], bool]


class Agent:
    """Minimal model-decides / we-execute loop.

    Still to build (see CLAUDE.md roadmap): richer guardrails, streaming,
    tracing, memory.
    """

    def __init__(self, registry: Registry, client=None, model: str | None = None,
                 max_rounds: int | None = None, temperature: float | None = None,
                 approve: ApproveFn | None = None) -> None:
        self.registry = registry
        self.client = client or make_client()
        self.model = model or settings.model
        self.max_rounds = max_rounds or settings.max_rounds
        self.temperature = settings.temperature if temperature is None else temperature
        self.approve = approve

    def run(self, user_text: str, history: list | None = None) -> tuple[str, list]:
        """Returns (final_text, updated_history)."""
        messages = list(history or [])
        messages.append({"role": "user", "content": user_text})

        for _ in range(self.max_rounds):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.registry.schemas() or None,
                temperature=self.temperature,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or "", messages
            messages.append(msg)  # the assistant tool-call turn
            for tc in msg.tool_calls:
                result = self._execute(tc)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result)})
        return "(max rounds reached)", messages

    def _execute(self, tc) -> dict:
        tool = self.registry.get(tc.function.name)
        if tool is None:
            return {"error": f"unknown tool: {tc.function.name}"}
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            return {"error": "could not parse tool arguments"}
        # Reject (never repair) bad args, before approval: the model gets the
        # errors plus the schema and can retry next round.
        if errors := tool.validate(args):
            return {"error": f"invalid arguments for tool '{tool.name}'",
                    "details": errors, "expected": tool.parameters}
        if tool.dangerous and self.approve and not self.approve(tool, args):
            return {"error": "tool call not approved"}
        try:
            return tool.func(**args)
        except Exception as e:  # noqa: BLE001 — surface any tool error to the model
            return {"error": f"{type(e).__name__}: {e}"}
