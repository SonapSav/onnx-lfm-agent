"""Unit tests that need no running server."""

from onnx_lfm_agent.tools import Registry


def test_registry_schema_and_dispatch():
    r = Registry()

    @r.tool(description="echo x",
            parameters={"type": "object",
                        "properties": {"x": {"type": "string"}},
                        "required": ["x"]})
    def echo(x):
        return {"x": x}

    assert len(r) == 1
    schema = r.schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["parameters"]["required"] == ["x"]
    assert r.get("echo").func(x="hi") == {"x": "hi"}
    assert r.get("missing") is None
