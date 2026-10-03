"""Test doubles shaped like the bits of the OpenAI client the agent touches."""

from types import SimpleNamespace as NS


def tool_call(arguments: str, name: str = "add", id: str = "call_1"):
    return NS(id=id, function=NS(name=name, arguments=arguments))


def reply(content=None, *tool_calls):
    """An assistant message: text, or tool calls (content None)."""
    return NS(content=content, tool_calls=list(tool_calls) or None)


class FakeClient:
    """Replays scripted assistant messages and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.sent.append({**kw, "messages": list(kw["messages"])})  # snapshot; run() keeps appending
        return NS(choices=[NS(message=self.replies.pop(0))])
