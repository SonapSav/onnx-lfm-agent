from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

from jsonschema import Draft202012Validator

# Id of the Agent.run() in progress, so tools can tell "earlier in this
# conversation's turn" from other requests (the HTTP service shares tools).
CURRENT_RUN: ContextVar[str | None] = ContextVar("lfm_agent_run", default=None)
# The user's message for that run: harness-driven tools decide against what the
# user asked, not the model's paraphrase of it in tool arguments.
CURRENT_REQUEST: ContextVar[str | None] = ContextVar("lfm_agent_request", default=None)

MAX_ARG_ERRORS = 5  # keep error feedback short; the 1.2B model drowns in long lists

# allow: run without asking. ask: run only if the approver says yes (denied when
# there is no approver, e.g. the HTTP service). deny: never run.
POLICIES = ("allow", "ask", "deny")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the arguments
    func: Callable[..., Any]
    dangerous: bool = False  # side-effecting -> defaults to policy "ask"
    policy: str | None = None  # None -> "ask" if dangerous else "allow"
    # Optional: text shown to a human approver (e.g. the diff a call would apply).
    preview: Callable[[dict], str] | None = None
    _validator: Draft202012Validator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = "ask" if self.dangerous else "allow"
        if self.policy not in POLICIES:
            raise ValueError(f"tool '{self.name}': policy must be one of {POLICIES}, "
                             f"got {self.policy!r}")
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


def parse_policy_spec(spec: str) -> dict[str, str]:
    """Parse "name=allow,other=deny" (LFM_TOOL_POLICY) into {name: policy}."""
    out: dict[str, str] = {}
    for item in filter(None, (part.strip() for part in spec.split(","))):
        name, sep, policy = (s.strip() for s in item.partition("="))
        if not sep or not name or policy not in POLICIES:
            raise ValueError(f"bad tool policy entry {item!r}; expected name=allow|ask|deny")
        out[name] = policy
    return out


class Registry:
    """Holds the tools the agent may call. Produces OpenAI `tools` schemas and
    dispatches by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # System-prompt lines contributed by toolsets (see prompts.py).
        self.guidance: list[str] = []

    def tool(self, *, description: str, parameters: dict, dangerous: bool = False,
             policy: str | None = None, preview: Callable[[dict], str] | None = None,
             name: str | None = None):
        """Decorator to register a function as a tool."""
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.add(Tool(name or fn.__name__, description, parameters, fn, dangerous,
                          policy, preview))
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
            for t in self
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())
