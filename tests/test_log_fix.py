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
        "database.pool_size", "logging.level", "none"]
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


def test_several_logs_the_model_picks_one(root):
    (root / "logs" / "other.log").write_text("ERROR disk full\n")
    tool, client = fix_tool(root, [
        reply(None, tool_call(json.dumps({"path": "logs/app.log"}), name="choose_log", id="l")),
        choose("server.request_timeout_s"), value(15)])
    out = tool.func(log_path="logs/missing.log")  # invented path: treated as omitted
    assert out["log"] == "logs/app.log" and out["proposal_id"] == "p1"
    shown = client.sent[0]["messages"][0]["content"]
    assert "logs/other.log:\n  ERROR disk full" in shown
    assert client.sent[0]["tools"][0]["function"]["parameters"]["properties"]["path"]["enum"] == [
        "logs/app.log", "logs/other.log"]


def test_none_means_no_proposal(root):
    tool, client = fix_tool(root, [choose("none")])
    out = tool.func()
    assert out["proposal_id"] is None and "no config change" in out["result"]
    assert "Do not propose" in out["next"]
    assert len(client.sent) == 1  # no value step


def test_several_configs_key_choice_spans_files(root):
    (root / "other.json").write_text('{"workers": 2}\n')
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "other")
    tool, client = fix_tool(root, [choose("other.json: workers"), value(4)])
    out = tool.func(config_path="nope.yaml")
    assert out["path"] == "other.json" and out["valid"]
    enum = client.sent[0]["tools"][0]["function"]["parameters"]["properties"]["key"]["enum"]
    assert "app.yaml: server.request_timeout_s" in enum and "other.json: workers" in enum


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


def test_extra_args_dropped_and_invented_ids_named(root):
    client = FakeClient([
        reply(None, tool_call(json.dumps({"key": "server.request_timeout_s", "value": 15}),
                              name="propose_config_change", id="p")),
        reply(None, tool_call(json.dumps({"proposal_id": "p123"}), name="apply_config_change", id="a")),
        reply(None, tool_call(json.dumps({"proposal_id": "p1", "proposal_description": "raise it"}),
                              name="apply_config_change", id="b")),
        reply("Applied."),
    ])
    agent = Agent(build_registry("workspace", workspace=str(root), client=client), client=client,
                  approve=lambda t, a: True, tool_policy="")
    result = agent.run("Set server.request_timeout_s to 15 and apply it.")

    invented, applied = result.steps[1], result.steps[2]
    assert invented.status == "tool_error"
    assert "p123" in invented.result["error"] and "pending in this conversation: p1" in invented.result["error"]
    assert applied.status == "executed" and applied.result["ignored_arguments"] == ["proposal_description"]
    assert "request_timeout_s: 15" in (root / "app.yaml").read_text()


@pytest.mark.parametrize("lines, current, schema, floor", [
    (["ERROR job 1832 rejected: payload 12.4 MB exceeds max_payload_mb=8"], 8,
     {"type": "number", "minimum": 1, "maximum": 100}, 12.4),   # the id 1832 is out of range
    (["ERROR timed out after 5s (request_timeout_s=5)"], 5,
     {"type": "number", "maximum": 120}, None),                  # nothing above the current
    (["ERROR payload 12.4 MB exceeds max_payload_mb=8"], 8, {"type": "number"}, None),  # unbounded
    (["ERROR payload 12.4 MB exceeds max_payload_mb=8"], "8", {"maximum": 100}, None),  # not numeric
])
def test_reported_above(lines, current, schema, floor):
    assert log_fix._reported_above(lines, current, schema) == floor


def test_value_below_what_the_log_reports_is_refused(root):
    (root / "logs" / "app.log").write_text(
        "2026-10-02 10:05:30 ERROR job 1832 rejected: payload 12.4 MB exceeds "
        "request_timeout_s=5\n")
    tool, client = fix_tool(root, [choose("server.request_timeout_s"), value(9), value(15)])
    out = tool.func()
    assert "+  request_timeout_s: 15" in out["diff"]
    error = json.loads(client.sent[-1]["messages"][-1]["content"])["error"]
    assert error == "the log reports 12.4 for server.request_timeout_s; choose at least 12.4"


def test_log_matching_the_request_is_picked_by_code(root):
    (root / "logs" / "worker.log").write_text("ERROR job rejected: payload too large\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "worker log")
    client = FakeClient([choose("server.request_timeout_s"), value(15)])  # no choose_log call
    r = Registry()
    wt.register(r, Workspace(root), client=client)
    from onnx_lfm_agent.tools import CURRENT_REQUEST
    token = CURRENT_REQUEST.set("The billing API keeps timing out. Check the logs and fix it.")
    try:
        out = r.get("propose_fix_from_logs").func()
    finally:
        CURRENT_REQUEST.reset(token)
    assert out["log"] == "logs/app.log"
    assert [c["tools"][0]["function"]["name"] for c in client.sent] == ["choose_setting", "set_value"]


def test_pending_proposal_blocks_proposals_for_other_files_too(root):
    (root / "other.json").write_text('{"workers": 2}\n')
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "other")
    client = FakeClient([
        reply(None, tool_call(json.dumps({"path": "app.yaml", "key": "server.port", "value": 81}),
                              name="propose_config_change", id="p")),
        reply(None, tool_call(json.dumps({"path": "other.json", "key": "workers", "value": 4}),
                              name="propose_config_change", id="q")),
        reply("Proposed."),
    ])
    agent = Agent(build_registry("workspace", workspace=str(root), client=client), client=client,
                  tool_policy="")
    result = agent.run("Set server.port to 81.")
    assert [s.status for s in result.steps] == ["executed", "tool_error"]
    assert "p1 is already proposed" in result.steps[1].result["error"]


def test_proposing_the_same_change_again_returns_the_pending_one(root):
    same = json.dumps({"key": "server.request_timeout_s", "value": 15})
    client = FakeClient([
        reply(None, tool_call(same, name="propose_config_change", id="p")),
        reply(None, tool_call(same, name="propose_config_change", id="q")),
        reply(None, tool_call('{"proposal_id": "p1"}', name="apply_config_change", id="a")),
        reply("Applied."),
    ])
    agent = Agent(build_registry("workspace", workspace=str(root), client=client), client=client,
                  approve=lambda t, a: True, tool_policy="")
    result = agent.run("Set server.request_timeout_s to 15, then apply it.")
    assert [s.status for s in result.steps] == ["executed", "executed", "executed"]
    assert result.steps[1].result["proposal_id"] == "p1"
    assert "already proposed as p1" in result.steps[1].result["note"]
