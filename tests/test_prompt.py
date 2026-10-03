"""System prompt: built from toolsets, sent first on every call, never stored."""

import pytest

from onnx_lfm_agent.agent import Agent
from onnx_lfm_agent.config import settings
from onnx_lfm_agent.prompts import BASE, build_system_prompt
from onnx_lfm_agent.toolsets import build_registry
from onnx_lfm_agent.workspace_tools import GUIDANCE

from fakes import FakeClient, reply, tool_call


def agent_with(replies, **kw):
    client = FakeClient(replies)
    return Agent(build_registry("demo"), client=client, tool_policy="", **kw), client


def test_prompt_is_base_plus_toolset_guidance():
    base = "\n".join(f"- {line}" for line in BASE)
    assert build_system_prompt(build_registry("demo")) == base
    # The workspace toolset deliberately contributes no guidance (see GUIDANCE).
    assert GUIDANCE == []
    assert build_system_prompt(build_registry("workspace", workspace="workspace")) == base
    r = build_registry("demo")
    r.guidance.append("Extra rule.")
    assert build_system_prompt(r) == base + "\n- Extra rule."


def test_off_by_default():
    agent, client = agent_with([reply("ok")])
    agent.run("hi")
    assert agent.system_prompt == ""
    assert client.sent[0]["messages"] == [{"role": "user", "content": "hi"}]


def test_sent_first_on_every_call_but_not_in_history():
    agent, client = agent_with([reply(None, tool_call('{"a": 1, "b": 2}')), reply("3")],
                               system_prompt="builtin")
    result = agent.run("1+2?")
    assert len(client.sent) == 2
    for call in client.sent:
        assert call["messages"][0] == {"role": "system", "content": agent.system_prompt}
        assert [m["role"] for m in call["messages"]].count("system") == 1
    assert all(m["role"] != "system" for m in result.messages)


def test_leading_system_message_in_history_is_replaced():
    agent, client = agent_with([reply("ok")], system_prompt="builtin")
    agent.run("hi", history=[{"role": "system", "content": "old prompt"},
                             {"role": "user", "content": "earlier"},
                             {"role": "assistant", "content": "sure"}])
    sent = client.sent[0]["messages"]
    assert [m.get("content") for m in sent] == [agent.system_prompt, "earlier", "sure", "hi"]


def test_explicit_override_and_disable():
    agent, client = agent_with([reply("ok")], system_prompt="Be terse.")
    agent.run("hi")
    assert client.sent[0]["messages"][0] == {"role": "system", "content": "Be terse."}

    agent, client = agent_with([reply("ok")], system_prompt="")
    agent.run("hi")
    assert client.sent[0]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("env_value, expected", [
    ("Custom.", "Custom."), ("", None),
    ("builtin", "\n".join(f"- {line}" for line in BASE)),
])
def test_env_setting(monkeypatch, env_value, expected):
    monkeypatch.setattr(settings, "system_prompt", env_value)
    agent, client = agent_with([reply("ok")])
    agent.run("hi")
    first = client.sent[0]["messages"][0]
    if expected is None:
        assert first == {"role": "user", "content": "hi"}
    else:
        assert first == {"role": "system", "content": expected}
