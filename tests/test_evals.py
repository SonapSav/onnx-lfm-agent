"""Scenario files (evals.load_scenarios): the expected end state is the check.
No model: the "run" is simulated by editing the seeded copy by hand."""

import json

import pytest

from onnx_lfm_agent.evals import load_scenarios, seeded_workspace


@pytest.fixture
def seed(tmp_path):
    root = tmp_path / "seed"
    (root / "svc").mkdir(parents=True)
    (root / "svc" / "a.yaml").write_text("pool:\n  size: 10\nmode: fast\n")
    (root / "b.json").write_text('{"items": [{"name": "x", "n": 1}]}\n')
    (root / "scenarios.json").write_text(json.dumps({"scenarios": [
        {"key": "S1", "name": "raise", "prompt": "p",
         "expect": {"changes": {"svc/a.yaml:pool.size": {"op": "gt", "value": 10}}}},
        {"key": "S2", "name": "nothing", "prompt": "p", "expect": {"changes": {}}},
    ]}))
    return root


def run(seed, edit):
    scenarios = load_scenarios(seed / "scenarios.json")
    with seeded_workspace(seed) as root:
        edit(root)
        return {k: s.check(None, root) for k, s in scenarios.items()}


def test_untouched_passes_only_the_no_change_scenario(seed):
    assert run(seed, lambda root: None) == {"S1": False, "S2": True}


def test_expected_change_passes(seed):
    def edit(root):
        (root / "svc" / "a.yaml").write_text("pool:\n  size: 20\nmode: fast\n")
    assert run(seed, edit) == {"S1": True, "S2": False}


@pytest.mark.parametrize("text", [
    "pool:\n  size: 5\nmode: fast\n",     # changed, but the wrong way
    "pool:\n  size: 20\nmode: slow\n",    # right change plus a stray one
    "pool:\n  size: '20'\nmode: fast\n",  # a string, not a number
])
def test_wrong_or_extra_changes_fail(seed, text):
    def edit(root):
        (root / "svc" / "a.yaml").write_text(text)
    assert run(seed, edit)["S1"] is False


def test_changes_in_other_files_are_seen(seed):
    def edit(root):
        (root / "b.json").write_text('{"items": [{"name": "x", "n": 2}]}\n')
    assert run(seed, edit) == {"S1": False, "S2": False}


def test_grading_files_never_reach_the_agent(seed):
    (seed / "DESIGN.md").write_text("answers\n")
    with seeded_workspace(seed) as root:
        assert not (root / "scenarios.json").exists() and not (root / "DESIGN.md").exists()
