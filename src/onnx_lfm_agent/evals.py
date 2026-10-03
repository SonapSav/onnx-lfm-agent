"""Live evaluation scenarios against a real model (used by scripts/eval_live.py
and the integration tests).

Each run gets a fresh temporary git workspace seeded from examples/workspace,
so evals never touch ./workspace and work on any machine.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from . import config_edit as ce
from .agent import Agent, RunResult
from .client import make_client
from .toolsets import build_registry

SEED_DIR = Path(__file__).resolve().parents[2] / "examples" / "workspace"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=eval", "-c",
                           "user.email=eval@localhost", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@contextmanager
def seeded_workspace() -> Iterator[Path]:
    """A throwaway git repo holding a copy of examples/workspace."""
    if not SEED_DIR.is_dir():
        raise FileNotFoundError(f"eval seed not found: {SEED_DIR} (run from a repo checkout)")
    with tempfile.TemporaryDirectory(prefix="lfm-eval-") as tmp:
        root = Path(tmp) / "ws"
        shutil.copytree(SEED_DIR, root)
        _git(root, "init", "-q", "-b", "main")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "seed")
        yield root


def _proposal_diffs(res: RunResult) -> list[str]:
    return [s.result["diff"] for s in res.steps
            if s.tool in ("propose_config_change", "propose_fix_from_logs")
            and isinstance(s.result, dict) and s.result.get("proposal_id")]


def _head_subject(root: Path) -> str:
    return _git(root, "log", "-1", "--format=%s")


def _ok_directed_apply(res: RunResult, root: Path) -> bool:
    return (_head_subject(root) == "agent: set server.request_timeout_s in app.yaml"
            and "request_timeout_s: 15" in (root / "app.yaml").read_text())


def _ok_directed_propose(res: RunResult, root: Path) -> bool:
    return (any("+  level: debug" in d for d in _proposal_diffs(res))
            and _head_subject(root) == "seed")


def _only_timeout_changed(root: Path) -> bool:
    """app.yaml differs from the seed in request_timeout_s at most: an open-ended
    run that also applied something else (seen: server.port=10) fails."""
    from .log_fix import leaves
    now = leaves(ce.parse((root / "app.yaml").read_text(), "yaml"))
    seed = leaves(ce.parse((SEED_DIR / "app.yaml").read_text(), "yaml"))
    return {k: v for k, v in now.items() if k != "server.request_timeout_s"} == \
        {k: v for k, v in seed.items() if k != "server.request_timeout_s"}


def _ok_open_ended(res: RunResult, root: Path) -> bool:
    if not _only_timeout_changed(root):
        return False
    for d in _proposal_diffs(res):
        for line in d.splitlines():
            if line.startswith("+  request_timeout_s:"):
                try:
                    if float(line.split(":", 1)[1].split("#")[0]) > 5:
                        return True
                except ValueError:
                    pass
    return False


def _ok_no_tools(res: RunResult, root: Path) -> bool:
    final = res.answer.split("</think>")[-1]  # thinking models: judge the answer only
    return not res.steps and "paris" in final.lower()


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    prompt: str
    check: Callable[[RunResult, Path], bool]


SCENARIOS = {s.key: s for s in [
    Scenario("A", "directed apply",
             "The logs show upstream timeouts at 5s. Propose changing "
             "server.request_timeout_s in app.yaml to 15, then apply it.", _ok_directed_apply),
    Scenario("B", "directed propose (must not apply)",
             "Propose setting logging.level in app.yaml to debug.", _ok_directed_propose),
    Scenario("C", "open-ended log -> config",
             "Check logs/app.log for errors and propose a config change in app.yaml "
             "that fixes them.", _ok_open_ended),
    Scenario("D", "no tools needed",
             "What is the capital of France?", _ok_no_tools),
]}


class TimedClient:
    """Wraps the OpenAI client to time model calls and count output tokens."""

    def __init__(self, inner=None) -> None:
        self.inner = inner or make_client()
        self.call_s: list[float] = []
        self.out_tokens: list[int] = []
        self.truncated = 0
        outer = self

        class _Completions:
            def create(self, **kw):
                t0 = time.perf_counter()
                r = outer.inner.chat.completions.create(**kw)
                outer.call_s.append(time.perf_counter() - t0)
                if r.usage:
                    outer.out_tokens.append(r.usage.completion_tokens)
                outer.truncated += r.choices[0].finish_reason == "length"
                return r

        self.chat = type("Chat", (), {"completions": _Completions()})()


@dataclass
class Outcome:
    passed: bool
    result: RunResult
    client: TimedClient
    tools: list[str] = field(default_factory=list)
    config_ok: bool = True  # app.yaml still parses and passes its schema after the run


def run_scenario(scenario: Scenario, system_prompt: str | None = None) -> Outcome:
    """One run in a fresh workspace; applies are auto-approved so the check sees
    what the model would do if you said yes."""
    with seeded_workspace() as root:
        client = TimedClient()
        agent = Agent(build_registry("workspace", workspace=str(root), client=client), client=client,
                      approve=lambda tool, args: True, tool_policy="",
                      system_prompt=system_prompt)
        res = agent.run(scenario.prompt)
        cfg = root / "app.yaml"
        try:
            config_ok = not ce.validate(ce.parse(cfg.read_text(), "yaml"), cfg)[0]
        except ce.ConfigError:
            config_ok = False
        return Outcome(scenario.check(res, root), res, client, [s.tool for s in res.steps],
                       config_ok)
