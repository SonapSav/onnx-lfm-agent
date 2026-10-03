"""Live evaluation scenarios against a real model (used by scripts/eval_live.py
and the integration tests).

Each run gets a fresh temporary git workspace seeded from examples/workspace,
so evals never touch ./workspace and work on any machine.
"""

from __future__ import annotations

import json
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

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
SEED_DIR = EXAMPLES / "workspace"  # one app.yaml + one log (scenarios A-D)
OPS_DIR = EXAMPLES / "workspace-ops"  # 3 configs (YAML/JSON, a list), 3 logs, decoys (E-H)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=eval", "-c",
                           "user.email=eval@localhost", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


# Grading material that lives next to a seed workspace but must never reach the
# agent (the first held-out run copied it in: answers visible, and scenarios.json
# was treated as a config file).
EVAL_ONLY = ("scenarios.json", "DESIGN.md")


@contextmanager
def seeded_workspace(seed: Path = SEED_DIR) -> Iterator[Path]:
    """A throwaway git repo holding a copy of `seed` (an examples/ workspace)."""
    if not seed.is_dir():
        raise FileNotFoundError(f"eval seed not found: {seed} (run from a repo checkout)")
    with tempfile.TemporaryDirectory(prefix="lfm-eval-") as tmp:
        root = Path(tmp) / "ws"
        shutil.copytree(seed, root, ignore=shutil.ignore_patterns(*EVAL_ONLY))
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


def _configs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix in ce.SUFFIXES
                  and not p.name.endswith(".schema.json") and p.name not in EVAL_ONLY
                  and ".git" not in p.parts)


def config_changes(root: Path, seed: Path) -> dict[str, tuple]:
    """{"file:key": (seed value, value now)} over every config file."""
    from .log_fix import leaves
    out = {}
    for f in _configs(seed):
        rel = f.relative_to(seed).as_posix()
        fmt = ce.config_format(f)
        before = leaves(ce.parse(f.read_text(), fmt))
        after = leaves(ce.parse((root / rel).read_text(), fmt))
        for k in before.keys() | after.keys():
            if before.get(k, "<missing>") != after.get(k, "<missing>"):
                out[f"{rel}:{k}"] = (before.get(k, "<missing>"), after.get(k, "<missing>"))
    return out


def _only_change(where: str, ok: Callable[[object], bool]) -> Callable[[RunResult, Path], bool]:
    """Passes iff exactly `where` ("file:key") changed, to a value `ok` accepts."""
    def check(res: RunResult, root: Path) -> bool:
        changes = config_changes(root, OPS_DIR)
        return set(changes) == {where} and ok(changes[where][1])
    return check


def _nothing_changed(res: RunResult, root: Path) -> bool:
    return not config_changes(root, OPS_DIR) and _head_subject(root) == "seed"


def _num(pred: Callable[[float], bool]) -> Callable[[object], bool]:
    return lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and pred(v)


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    prompt: str
    check: Callable[[RunResult, Path], bool]
    seed: Path = SEED_DIR


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
    # workspace-ops: several configs and logs, decoy errors. Each asks to apply,
    # and passes only on the exact end state of every config file.
    Scenario("E", "ops: log -> JSON limit",
             "Check logs/worker.log for errors and fix the config, then apply it.",
             _only_change("services/worker.json:limits.max_payload_mb", _num(lambda v: v >= 12.4)),
             OPS_DIR),
    Scenario("F", "ops: symptom only, pick log+config",
             "The API keeps returning 503 errors. Look at the logs and fix the config, "
             "then apply it.",
             _only_change("services/api.yaml:database.pool_size", _num(lambda v: v > 10)),
             OPS_DIR),
    Scenario("G", "ops: directed list key",
             "In services/worker.json set the concurrency of the reports queue to 4, "
             "then apply it.",
             _only_change("services/worker.json:queues.1.concurrency", lambda v: v == 4),
             OPS_DIR),
    Scenario("H", "ops: errors config can't fix",
             "Check logs/auth.log. If a config change would fix the errors, apply it.",
             _nothing_changed, OPS_DIR),
]}


OPS = {"eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "gt": lambda a, b: a > b,
       "ge": lambda a, b: a >= b, "lt": lambda a, b: a < b, "le": lambda a, b: a <= b}


def load_scenarios(path: Path) -> dict[str, Scenario]:
    """Scenarios from a JSON file next to its seed workspace (format: see
    examples/workspace-heldout/scenarios.json). Each passes only if exactly the
    listed "file:key"s changed, each meeting its {"op", "value"} condition;
    {} = nothing may change."""
    seed = path.parent
    out = {}
    for s in json.loads(path.read_text())["scenarios"]:
        expected = s["expect"]["changes"]

        def check(res: RunResult, root: Path, expected=expected, seed=seed) -> bool:
            changes = config_changes(root, seed)
            if set(changes) != set(expected):
                return False
            for where, cond in expected.items():
                new = changes[where][1]
                try:
                    if not OPS[cond["op"]](new, cond["value"]):
                        return False
                except TypeError:  # e.g. a string where a number was expected
                    return False
            return True

        out[s["key"]] = Scenario(s["key"], s["name"], s["prompt"], check, seed)
    return out


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
    config_ok: bool = True  # every config file still parses and passes its schema


def run_scenario(scenario: Scenario, system_prompt: str | None = None) -> Outcome:
    """One run in a fresh workspace; applies are auto-approved so the check sees
    what the model would do if you said yes."""
    with seeded_workspace(scenario.seed) as root:
        client = TimedClient()
        agent = Agent(build_registry("workspace", workspace=str(root), client=client), client=client,
                      approve=lambda tool, args: True, tool_policy="",
                      system_prompt=system_prompt)
        res = agent.run(scenario.prompt)
        try:
            config_ok = not any(ce.validate(ce.parse(f.read_text(), ce.config_format(f)), f)[0]
                                for f in _configs(root))
        except ce.ConfigError:
            config_ok = False
        return Outcome(scenario.check(res, root), res, client, [s.tool for s in res.steps],
                       config_ok)
