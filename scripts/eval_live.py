"""Live eval of the agent against a running onnx-lfm-api (real model).

Scenarios (src/onnx_lfm_agent/evals.py):
  A  "propose X, then apply it"      -> committed
  B  "propose X"                     -> valid proposal, NOT applied
  C  open-ended "check logs and fix" -> valid proposal raising the timeout
  D  "capital of France?"            -> answered without tools
  examples/workspace-ops (3 configs, 3 logs, decoy errors; each asks to apply,
  checked on the end state of every config file):
  E  "check worker.log and fix"      -> only limits.max_payload_mb, >= the 12.4 MB logged
  F  "the API returns 503s" (no file named) -> only database.pool_size in api.yaml, raised
  G  "set the reports queue's concurrency to 4" -> only queues.1.concurrency = 4
  H  "check auth.log; fix if config helps" (wrong passwords) -> nothing changed

Uses LFM_URL / LFM_API_KEY / LFM_TEMPERATURE from the environment or ./.env.
Each run uses a fresh temp git workspace; applies are auto-approved.

  .venv/bin/python scripts/eval_live.py                 # all scenarios, 6 runs, LFM_SYSTEM_PROMPT (default off)
  .venv/bin/python scripts/eval_live.py -n 3 -s A,B     # quicker
  .venv/bin/python scripts/eval_live.py --compare-prompt  # system prompt off vs built-in
  LFM_URL=http://gpu-pc:8383/v1 .venv/bin/python scripts/eval_live.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onnx_lfm_agent.config import settings  # noqa: E402
from onnx_lfm_agent.evals import SCENARIOS, run_scenario  # noqa: E402
from onnx_lfm_agent.prompts import BUILTIN  # noqa: E402


def _mode(prompt: str) -> str:
    return "off" if not prompt else "on" if prompt == BUILTIN else "custom"

SHORT = {"propose_config_change": "propose", "apply_config_change": "apply", "propose_fix_from_logs": "fix",
         "search_files": "search", "read_file": "read", "list_files": "list"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--runs", type=int, default=6, help="runs per scenario (default 6)")
    ap.add_argument("-s", "--scenarios", default=",".join(SCENARIOS), help="comma-separated keys")
    ap.add_argument("--compare-prompt", action="store_true",
                    help="run each scenario with the system prompt off, then on")
    ap.add_argument("--json", type=Path, help="also write results here")
    args = ap.parse_args()

    keys = [k.strip().upper() for k in args.scenarios.split(",") if k.strip()]
    if unknown := [k for k in keys if k not in SCENARIOS]:
        sys.exit(f"unknown scenario(s) {unknown}; choose from {list(SCENARIOS)}")
    modes = [("off", ""), ("on", BUILTIN)] if args.compare_prompt else [(_mode(settings.system_prompt), None)]

    print(f"model API: {settings.url}  temperature: {settings.temperature}  runs: {args.runs}", flush=True)
    rows = []
    for mode, system_prompt in modes:
        for key in keys:
            sc = SCENARIOS[key]
            outcomes = [run_scenario(sc, system_prompt) for _ in range(args.runs)]
            calls = [t for o in outcomes for t in o.client.call_s]
            toks = [t for o in outcomes for t in o.client.out_tokens]
            row = {
                "prompt": mode, "scenario": key, "name": sc.name,
                "passed": sum(o.passed for o in outcomes), "runs": args.runs,
                "avg_call_s": round(statistics.mean(calls), 1) if calls else None,
                "avg_out_tokens": round(statistics.mean(toks)) if toks else None,
                "truncated_calls": sum(o.client.truncated for o in outcomes),
                "tool_paths": [">".join(SHORT.get(t, t) for t in o.tools) or "-" for o in outcomes],
            }
            rows.append(row)
            print(f"[prompt {mode:3}] {key} {sc.name:34} {row['passed']}/{args.runs}  "
                  f"call {row['avg_call_s']}s  out {row['avg_out_tokens']} tok  "
                  f"cut {row['truncated_calls']}  | {' | '.join(row['tool_paths'])}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
