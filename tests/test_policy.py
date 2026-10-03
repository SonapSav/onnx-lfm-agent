"""Per-tool allow / ask / deny policy. No server needed."""

import logging

import pytest

from onnx_lfm_agent.agent import Agent
from onnx_lfm_agent.tools import Registry, Tool, parse_policy_spec

from fakes import FakeClient, reply, tool_call

NO_ARGS = {"type": "object", "properties": {}}


def make_registry(ran: list) -> Registry:
    r = Registry()

    @r.tool(description="read", parameters=NO_ARGS)
    def read_thing():
        ran.append("read_thing")
        return {"ok": True}

    @r.tool(description="write", parameters=NO_ARGS, dangerous=True)
    def write_thing():
        ran.append("write_thing")
        return {"ok": True}

    @r.tool(description="nuke", parameters=NO_ARGS, policy="deny")
    def nuke():
        ran.append("nuke")
        return {"ok": True}

    return r


def agent_for(registry, approve=None, tool_policy="", client=None) -> Agent:
    return Agent(registry, client=client or object(), approve=approve, tool_policy=tool_policy)


def test_defaults():
    agent = agent_for(make_registry([]))
    assert agent.policies == {"read_thing": "allow", "write_thing": "ask", "nuke": "deny"}


def test_bad_tool_policy_value_rejected():
    with pytest.raises(ValueError, match="policy must be one of"):
        Tool("t", "t", NO_ARGS, lambda: {}, policy="maybe")


@pytest.mark.parametrize("spec, expected", [
    ("", {}),
    ("a=allow", {"a": "allow"}),
    (" a = deny , b=ask ,", {"a": "deny", "b": "ask"}),
])
def test_parse_policy_spec(spec, expected):
    assert parse_policy_spec(spec) == expected


@pytest.mark.parametrize("spec", ["a", "a=", "=allow", "a=maybe"])
def test_parse_policy_spec_rejects_garbage(spec):
    with pytest.raises(ValueError, match="bad tool policy entry"):
        parse_policy_spec(spec)


def test_override_unknown_tool_rejected():
    with pytest.raises(ValueError, match="unknown tool 'nope'"):
        agent_for(make_registry([]), tool_policy="nope=allow")


def test_override_loosening_dangerous_tool_warns(caplog):
    with caplog.at_level(logging.WARNING):
        agent = agent_for(make_registry([]), tool_policy="write_thing=allow,read_thing=deny")
    assert agent.policies["write_thing"] == "allow"
    assert agent.policies["read_thing"] == "deny"
    assert "dangerous tool 'write_thing'" in caplog.text


def test_allow_runs_without_asking():
    ran, asked = [], []
    agent = agent_for(make_registry(ran), approve=lambda t, a: asked.append(t.name) or False)
    step = agent._execute(tool_call("{}", name="read_thing"))
    assert (step.status, ran, asked) == ("executed", ["read_thing"], [])


def test_deny_never_asks():
    ran, asked = [], []
    agent = agent_for(make_registry(ran), approve=lambda t, a: asked.append(t.name) or True)
    step = agent._execute(tool_call("{}", name="nuke"))
    assert step.status == "denied"
    assert step.result == {"error": "tool 'nuke' is denied by policy"}
    assert (ran, asked) == ([], [])


@pytest.mark.parametrize("answer, status, ran_expected", [
    (True, "executed", ["write_thing"]),
    (False, "denied", []),
])
def test_ask_uses_approver(answer, status, ran_expected):
    ran = []
    agent = agent_for(make_registry(ran), approve=lambda t, a: answer)
    step = agent._execute(tool_call("{}", name="write_thing"))
    assert step.status == status
    assert ran == ran_expected


def test_ask_without_approver_is_denied():
    ran = []
    step = agent_for(make_registry(ran))._execute(tool_call("{}", name="write_thing"))
    assert step.status == "denied"
    assert "no approver is available" in step.result["error"]
    assert ran == []


def test_denial_is_fed_back_to_model():
    client = FakeClient([reply(None, tool_call("{}", name="nuke", id="n1")), reply("can't")])
    result = agent_for(make_registry([]), client=client).run("nuke it")
    assert result.answer == "can't"
    tool_msg = client.sent[1]["messages"][-1]
    assert tool_msg["tool_call_id"] == "n1"
    assert "denied by policy" in tool_msg["content"]
