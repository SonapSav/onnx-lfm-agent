from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .client import make_client
from .config import settings
from .tools import Registry, Tool, parse_policy_spec

log = logging.getLogger(__name__)

# Return True to allow a tool call. Called only for tools whose policy is
# "ask"; "allow" tools run and "deny" tools are refused without asking.
ApproveFn = Callable[[Tool, dict], bool]


@dataclass
class Step:
    """One tool call the model asked for, and what the harness did with it.

    status: executed | tool_error | invalid_args | denied | unknown_tool | bad_json
    """
    tool: str
    args: Any
    status: str
    result: Any

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    answer: str
    messages: list = field(default_factory=list)  # plain dicts; JSON-serializable
    steps: list[Step] = field(default_factory=list)


class Agent:
    """Minimal model-decides / we-execute loop.

    Still to build (see CLAUDE.md roadmap): propose -> validate -> apply for
    config-changing tools, streaming, tracing, memory.
    """

    def __init__(self, registry: Registry, client=None, model: str | None = None,
                 max_rounds: int | None = None, temperature: float | None = None,
                 approve: ApproveFn | None = None, tool_policy: str | None = None) -> None:
        self.registry = registry
        self.client = client or make_client()
        self.model = model or settings.model
        self.max_rounds = max_rounds or settings.max_rounds
        self.temperature = settings.temperature if temperature is None else temperature
        self.approve = approve
        self.policies = self._resolve_policies(
            settings.tool_policy if tool_policy is None else tool_policy)

    def _resolve_policies(self, spec: str) -> dict[str, str]:
        """Tool defaults, overlaid with the LFM_TOOL_POLICY-style `spec`."""
        overrides = parse_policy_spec(spec)
        policies = {}
        for name in overrides:
            if self.registry.get(name) is None:
                raise ValueError(f"tool policy names unknown tool {name!r}")
        for tool in self.registry:
            policy = overrides.get(tool.name, tool.policy)
            if tool.dangerous and policy == "allow":
                log.warning("dangerous tool %r is set to 'allow': it will run without approval",
                            tool.name)
            policies[tool.name] = policy
        return policies

    def run(self, user_text: str, history: list | None = None) -> RunResult:
        messages = list(history or [])
        messages.append({"role": "user", "content": user_text})
        steps: list[Step] = []

        for _ in range(self.max_rounds):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.registry.schemas() or None,
                temperature=self.temperature,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                answer = msg.content or ""
                # Keep the answer in history so the next turn can see it.
                messages.append({"role": "assistant", "content": answer})
                return RunResult(answer, messages, steps)
            messages.append(_assistant_turn(msg))
            for tc in msg.tool_calls:
                step = self._execute(tc)
                log.info("tool %s(%s) -> %s", step.tool, step.args, step.status)
                steps.append(step)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(step.result)})
        return RunResult("(max rounds reached)", messages, steps)

    def _execute(self, tc) -> Step:
        name, raw = tc.function.name, tc.function.arguments
        tool = self.registry.get(name)
        if tool is None:
            return Step(name, raw, "unknown_tool", {"error": f"unknown tool: {name}"})
        try:
            args = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return Step(name, raw, "bad_json", {"error": "could not parse tool arguments"})
        # Reject (never repair) bad args, before the policy check: the model gets
        # the errors plus the schema and can retry next round.
        if errors := tool.validate(args):
            return Step(name, args, "invalid_args", {
                "error": f"invalid arguments for tool '{name}'",
                "details": errors, "expected": tool.parameters})
        if denial := self._policy_denial(tool, args):
            return Step(name, args, "denied", {"error": denial})
        try:
            return Step(name, args, "executed", tool.func(**args))
        except Exception as e:  # noqa: BLE001 — surface any tool error to the model
            return Step(name, args, "tool_error", {"error": f"{type(e).__name__}: {e}"})

    def _policy_denial(self, tool: Tool, args: dict) -> str | None:
        """None if the call may run, else the reason it may not."""
        policy = self.policies[tool.name]
        if policy == "allow":
            return None
        if policy == "deny":
            return f"tool '{tool.name}' is denied by policy"
        if self.approve is None:
            return f"tool '{tool.name}' needs approval and no approver is available"
        if not self.approve(tool, args):
            return f"tool '{tool.name}' was not approved"
        return None


def _assistant_turn(msg) -> dict:
    """The assistant tool-call message as a plain dict (history must be JSON)."""
    return {
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ],
    }
