"""propose_fix_from_logs: code finds the evidence and lists the real keys, the
model only picks a key and a value. Temp git workspace from examples/workspace;
the model is a FakeClient."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from onnx_lfm_agent import log_fix
from onnx_lfm_agent import workspace_tools as wt
from onnx_lfm_agent.agent import Agent
from onnx_lfm_agent.toolsets import build_registry
from onnx_lfm_agent.tools import Registry
from onnx_lfm_agent.workspace import Workspace, WorkspaceError

from fakes import FakeClient, reply, tool_call

SEED = Path(__file__).parent.parent / "examples" / "workspace"


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=Human",
                           "-c", "user.email=human@example.com", *args],
                          check=True, capture_output=True, text=True).stdout


@pytest.fixture
def root(tmp_path):
    ws = tmp_path / "ws"
    shutil.copytree(SEED, ws)
    git(ws, "init", "-q", "-b", "main")
    git(ws, "add", "-A")
    git(ws, "commit", "-q", "-m", "seed")
    return ws


def fix_tool(root, replies):
    client = FakeClient(replies)
    r = Registry()
    wt.register(r, Workspace(root), client=client)
    return r.get("propose_fix_from_logs"), client


def choose(key):
    return reply(None, tool_call(json.dumps({"key": key}), name="choose_setting", id="k"))


def value(v):
    return reply(None, tool_call(json.dumps({"value": v}), name="set_value", id="v"))


def test_evidence_dedupes_and_strips_timestamps():
    text = ("2026-10-02 14:00:01 INFO  started\n"
            "2026-10-02 14:05:40 ERROR upstream timed out after 5s\n"
            "2026-10-02T14:09:18Z ERROR upstream timed out after 5s\n"
            "2026-10-02 14:10:00 WARN  pool nearly full\n"
            "Traceback (most recent call last):\n")
    assert log_fix.evidence(text) == ["2x ERROR upstream timed out after 5s",
                                      "WARN  pool nearly full",
                                      "Traceback (most recent call last):"]


def test_proposes_the_chosen_key_and_value(root):
    tool, client = fix_tool(root, [choose("server.request_timeout_s"), value(15)])
    out = tool.func()  # log and config inferred: the only .log / config file

    assert out["log"] == "logs/app.log" and out["path"] == "app.yaml"
    assert out["evidence"][0] == "3x ERROR upstream billing API timed out after 5s (request_timeout_s=5)"
    assert out["valid"] and out["proposal_id"] == "p1"
    assert "+  request_timeout_s: 15" in out["diff"]
    assert git(root, "status", "--porcelain") == ""  # proposing writes nothing

    key_prompt = client.sent[0]["messages"][0]["content"]
    assert "server.request_timeout_s = 5  # number, 1..120; upstream calls to the billing API" in key_prompt
    assert client.sent[0]["tools"][0]["function"]["parameters"]["properties"]["key"]["enum"] == [
        "server.host", "server.port", "server.request_timeout_s", "database.url",
        "database.pool_size", "logging.level"]
    value_param = client.sent[1]["tools"][0]["function"]["parameters"]["properties"]["value"]
    assert value_param == {"type": "number", "minimum": 1, "maximum": 120, "description": "New value"}


def test_bad_choices_are_fed_back_and_retried(root):
    tool, client = fix_tool(root, [
        reply("I think the timeout."),           # no tool call -> asked again
        choose("server.timeout"),                # not one of the keys -> enum error
        choose("server.request_timeout_s"),
        value(5),                                # unchanged -> refused
        value(500),                              # over the schema maximum -> refused
        value(15),
    ])
    out = tool.func(log_path="logs/app.log", config_path="app.yaml")

    assert "+  request_timeout_s: 15" in out["diff"]
    def errors(call):  # each call carries its decision's whole conversation so far
        return [json.loads(m["content"])["error"] for m in call["messages"] if m["role"] == "tool"]
    key_step, value_step = client.sent[2], client.sent[-1]
    assert key_step["messages"][2]["content"] == "Answer by calling choose_setting."
    assert len(errors(key_step)) == 1 and "is not one of" in errors(key_step)[0]
    assert "already 5" in errors(value_step)[0]
    assert "500 is greater than the maximum of 120" in errors(value_step)[1]


def test_gives_up_after_max_tries(root):
    tool, _ = fix_tool(root, [choose("server.request_timeout_s")] + [value(5)] * log_fix.MAX_TRIES)
    with pytest.raises(WorkspaceError, match="valid set_value choice"):
        tool.func()


def test_no_errors_in_log_means_no_model_call(root):
    (root / "logs" / "app.log").write_text("2026-10-02 14:00:01 INFO  all good\n")
    tool, client = fix_tool(root, [])
    out = tool.func()
    assert out["proposal_id"] is None and "nothing to fix" in out["result"]
    assert client.sent == []


def test_ambiguous_log_file_is_refused_with_candidates(root):
    (root / "logs" / "other.log").write_text("ERROR x\n")
    tool, _ = fix_tool(root, [])
    with pytest.raises(WorkspaceError, match="which log file.*logs/app.log, logs/other.log"):
        tool.func()


def test_enum_key_gets_an_enum_value(root):
    tool, client = fix_tool(root, [choose("logging.level"), value("debug")])
    out = tool.func()
    assert "+  level: debug" in out["diff"]
    value_param = client.sent[1]["tools"][0]["function"]["parameters"]["properties"]["value"]
    assert value_param["enum"] == ["debug", "info", "warning", "error"]


def test_open_ended_request_end_to_end(root):
    """The outer model only has to pick the workflow tool, then apply."""
    client = FakeClient([
        reply(None, tool_call("{}", name="propose_fix_from_logs", id="f")),
        choose("server.request_timeout_s"), value(15),         # the workflow's own calls
        reply(None, tool_call('{"proposal_id": "p1"}', name="apply_config_change", id="a")),
        reply("Raised the timeout to 15."),
    ])
    agent = Agent(build_registry("workspace", workspace=str(root), client=client), client=client,
                  approve=lambda t, a: True, tool_policy="")
    result = agent.run("Check logs/app.log for errors and fix the config.")

    assert [s.status for s in result.steps] == ["executed", "executed"]
    assert "request_timeout_s: 15" in (root / "app.yaml").read_text()


def test_refused_when_this_run_already_proposed_for_the_file(root):
    """Directed "set X to 15, then apply": a stray call to the workflow must not
    produce a second proposal for the model to apply instead."""
    client = FakeClient([
        reply(None, tool_call(json.dumps({"key": "server.request_timeout_s", "value": 15}),
                              name="propose_config_change", id="p")),
        reply(None, tool_call("{}", name="propose_fix_from_logs", id="f")),
        reply(None, tool_call('{"proposal_id": "p1"}', name="apply_config_change", id="a")),
        reply("Applied."),
    ])
    agent = Agent(build_registry("workspace", workspace=str(root), client=client), client=client,
                  approve=lambda t, a: True, tool_policy="")
    result = agent.run("Set server.request_timeout_s to 15, the logs show timeouts. Then apply it.")

    assert [s.status for s in result.steps] == ["executed", "tool_error", "executed"]
    assert "p1 is already proposed in this conversation" in result.steps[1].result["error"]
    assert "request_timeout_s: 15" in (root / "app.yaml").read_text()


def test_a_new_run_is_not_blocked_by_an_earlier_runs_proposal(root):
    client = FakeClient([
        reply(None, tool_call(json.dumps({"key": "logging.level", "value": "debug"}),
                              name="propose_config_change", id="p")),
        reply("Proposed p1."),
        reply(None, tool_call("{}", name="propose_fix_from_logs", id="f")),
        choose("server.request_timeout_s"), value(15),
        reply("Proposed p2."),
    ])
    agent = Agent(build_registry("workspace", workspace=str(root), client=client), client=client,
                  tool_policy="")
    agent.run("Propose logging.level debug.")
    second = agent.run("Check the logs and propose a fix.")
    assert second.steps[0].status == "executed" and second.steps[0].result["proposal_id"] == "p2"
