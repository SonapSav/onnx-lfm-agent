"""Tool-argument validation in Agent._execute. No server needed."""

import json

import pytest
from jsonschema.exceptions import SchemaError

from onnx_lfm_agent.agent import Agent
from onnx_lfm_agent.tools import Registry, Tool

from fakes import FakeClient, reply, tool_call

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
    "required": ["a", "b"],
}


def make_registry(calls: list, dangerous: bool = False) -> Registry:
    r = Registry()

    @r.tool(description="Add two numbers.", parameters=ADD_SCHEMA, dangerous=dangerous)
    def add(a, b):
        calls.append((a, b))
        return {"sum": a + b}

    return r


def agent_for(registry, client=None, approve=None) -> Agent:
    # tool_policy="" so a local LFM_TOOL_POLICY can't leak into tests.
    return Agent(registry, client=client or object(), approve=approve, tool_policy="")


def test_valid_args_execute():
    calls = []
    out = agent_for(make_registry(calls))._execute(tool_call('{"a": 2, "b": 2}')).result
    assert out == {"sum": 4}
    assert calls == [(2, 2)]


@pytest.mark.parametrize("arguments, expected_detail", [
    ('{"a": 2}', "'b' is a required property"),
    ('{"a": "two", "b": 2}', "a: 'two' is not of type 'number'"),
    ('[2, 2]', "[2, 2] is not of type 'object'"),
    ('"2+2"', "'2+2' is not of type 'object'"),
])
def test_invalid_args_rejected_without_calling(arguments, expected_detail):
    calls = []
    out = agent_for(make_registry(calls))._execute(tool_call(arguments)).result
    assert out["error"] == "invalid arguments for tool 'add'"
    assert expected_detail in out["details"]
    assert out["expected"] == ADD_SCHEMA
    assert calls == []


def test_invalid_args_skip_approval():
    asked = []
    agent = agent_for(make_registry([], dangerous=True),
                      approve=lambda tool, args: asked.append(args) or True)
    out = agent._execute(tool_call('{"a": 1}')).result
    assert "details" in out
    assert asked == []


def test_errors_are_capped():
    schema = {"type": "object",
              "properties": {k: {"type": "integer"} for k in "abcdefgh"},
              "required": list("abcdefgh")}
    tool = Tool("t", "t", schema, lambda **kw: kw)
    assert len(tool.validate({})) == 5


def test_bad_schema_rejected_at_registration():
    r = Registry()
    with pytest.raises(SchemaError):
        @r.tool(description="broken", parameters={"type": "objekt"})
        def broken():
            return {}


def test_run_feeds_validation_error_back_and_model_recovers():
    calls = []
    client = FakeClient([
        reply(None, tool_call('{"a": 2}', id="bad")),
        reply(None, tool_call('{"a": 2, "b": 2}', id="good")),
        reply("4"),
    ])
    result = agent_for(make_registry(calls), client=client).run("2+2?")

    assert result.answer == "4"
    assert calls == [(2, 2)]  # only the valid call executed
    assert [s.status for s in result.steps] == ["invalid_args", "executed"]
    bad_result = next(m for m in result.messages if m.get("tool_call_id") == "bad")
    assert "'b' is a required property" in json.loads(bad_result["content"])["details"]
    json.dumps(result.messages)  # history is plain JSON, ready for the HTTP service
