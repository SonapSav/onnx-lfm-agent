from __future__ import annotations

import argparse

from .agent import Agent
from .example_tools import registry
from .tools import Tool


def _approve(tool: Tool, args: dict) -> bool:
    print(f"  [approve] {tool.name}({args})  — this tool is marked dangerous")
    try:
        return input("  run it? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="onnx-lfm-agent — tool-calling REPL")
    ap.add_argument("prompt", nargs="*", help="one-shot prompt (omit for an interactive REPL)")
    args = ap.parse_args()

    agent = Agent(registry, approve=_approve)

    if args.prompt:
        answer, _ = agent.run(" ".join(args.prompt))
        print(answer)
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
        answer, history = agent.run(user, history)
        print("bot>", answer)


if __name__ == "__main__":
    main()
