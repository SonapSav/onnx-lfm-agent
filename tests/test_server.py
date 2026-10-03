"""HTTP service, with a fake model client. No running API needed."""

from types import SimpleNamespace as NS

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from onnx_lfm_agent.agent import Agent
from onnx_lfm_agent.server import create_app
from onnx_lfm_agent.tools import Registry

from fakes import FakeClient, reply, tool_call

KEY = "test-key"
NO_ARGS = {"type": "object", "properties": {}}


def make_client(replies) -> TestClient:
    r = Registry()

    @r.tool(description="read", parameters=NO_ARGS)
    def read_thing():
        return {"value": 42}

    @r.tool(description="write", parameters=NO_ARGS, dangerous=True)
    def write_thing():
        raise AssertionError("must not run without approval")

    agent = Agent(r, client=FakeClient(replies), tool_policy="")
    return TestClient(create_app(agent=agent, api_key=KEY))


def test_refuses_to_start_without_key():
    with pytest.raises(RuntimeError, match="LFM_AGENT_API_KEY"):
        create_app(agent=Agent(Registry(), client=object(), tool_policy=""), api_key="")


def test_health_needs_no_auth():
    resp = make_client([]).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "tools": {"read_thing": "allow", "write_thing": "ask"}}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"Authorization": "Bearer wrong"}])
def test_run_rejects_bad_auth(headers):
    resp = make_client([]).post("/run", json={"prompt": "hi"}, headers=headers)
    assert resp.status_code == 401


@pytest.mark.parametrize("headers", [{"X-API-Key": KEY}, {"Authorization": f"Bearer {KEY}"}])
def test_run_accepts_either_auth_header(headers):
    resp = make_client([reply("hello")]).post("/run", json={"prompt": "hi"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["answer"] == "hello"


def test_run_returns_steps_and_denies_ask_tools():
    client = make_client([
        reply(None, tool_call("{}", name="read_thing", id="r"),
              tool_call("{}", name="write_thing", id="w")),
        reply("value is 42; could not write"),
    ])
    body = client.post("/run", json={"prompt": "read then write"},
                       headers={"X-API-Key": KEY}).json()

    assert body["answer"] == "value is 42; could not write"
    assert [(s["tool"], s["status"]) for s in body["steps"]] == [
        ("read_thing", "executed"), ("write_thing", "denied")]
    assert "no approver" in body["steps"][1]["result"]["error"]
    assert [m["role"] for m in body["history"]] == [
        "user", "assistant", "tool", "tool", "assistant"]


def test_run_continues_from_history():
    client = make_client([reply("first"), reply("second")])
    first = client.post("/run", json={"prompt": "one"}, headers={"X-API-Key": KEY}).json()
    second = client.post("/run", json={"prompt": "two", "history": first["history"]},
                         headers={"X-API-Key": KEY}).json()
    assert second["answer"] == "second"
    assert [m["content"] for m in second["history"]] == ["one", "first", "two", "second"]


def test_model_api_failure_is_502():
    def fail(**kw):
        raise openai.APIConnectionError(request=httpx.Request("POST", "http://api/v1"))

    broken = NS(chat=NS(completions=NS(create=fail)))
    agent = Agent(Registry(), client=broken, tool_policy="")
    client = TestClient(create_app(agent=agent, api_key=KEY))
    resp = client.post("/run", json={"prompt": "hi"}, headers={"X-API-Key": KEY})
    assert resp.status_code == 502
