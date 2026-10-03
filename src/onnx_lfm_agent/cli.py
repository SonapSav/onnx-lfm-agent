from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .config import settings
from .toolsets import build_registry
from .tools import Tool
from .workspace import Workspace, WorkspaceError
from .workspace_tools import last_change, rollback_last_change


def _approve(tool: Tool, args: dict) -> bool:
    print(f"  [approve] {tool.name}({args})  — this tool's policy is 'ask'")
    if tool.preview:
        print("\n".join(f"    {line}" for line in tool.preview(args).splitlines()))
    return _confirm("  run it?")


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def _rollback(assume_yes: bool) -> int:
    """Operator-only undo of the agent's latest applied config change."""
    ws = Workspace(settings.workspace)
    try:
        head = last_change(ws)
        if not head["by_agent"]:
            print(f"latest commit {head['commit']} ({head['subject']!r}) was not made by "
                  "the agent; nothing to roll back", file=sys.stderr)
            return 1
        print(f"will revert {head['commit']}: {head['subject']}")
        if not assume_yes and not _confirm("revert it?"):
            print("aborted")
            return 1
        out = rollback_last_change(ws)
    except WorkspaceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"reverted {out['reverted']} -> new commit {out['commit']}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="onnx-lfm-agent — tool-calling REPL")
    ap.add_argument("prompt", nargs="*", help="one-shot prompt (omit for an interactive REPL)")
    ap.add_argument("--rollback", action="store_true",
                    help="revert the agent's latest applied config change in the workspace")
    ap.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    args = ap.parse_args()

    if args.rollback:
        sys.exit(_rollback(args.yes))

    registry = build_registry()
    agent = Agent(registry, approve=_approve)

    if args.prompt:
        print(agent.run(" ".join(args.prompt)).answer)
        return

    print(f"onnx-lfm-agent | {len(registry)} tools loaded | /exit to quit")
    history: list = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            break
        result = agent.run(user, history)
        history = result.messages
        print("bot>", result.answer)


if __name__ == "__main__":
    main()
