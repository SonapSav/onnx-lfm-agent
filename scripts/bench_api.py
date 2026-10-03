"""Benchmark onnx-lfm-api speed: generation tok/s and a tool-calling request.

Two modes:
  * containers (default): for each --config, start a throwaway container of the
    API image (models mounted read-only, HF offline) on 127.0.0.1:8390, measure,
    remove it. The production API is never touched.
  * --url: measure an API that is already running (e.g. on a Jetson).

Workloads (temperature 0, median of --reps):
  generate  ~128 tokens of prose               -> decode speed (tok/s)
  agent     the agent's tool schemas + a request -> prompt-processing heavy

Examples:
  # CPU: quant x threads (laptop result: q4:6 best)
  .venv/bin/python scripts/bench_api.py --configs q4:0,q4:6,q4f16:6
  # CUDA PC (image built from the API repo's Dockerfile.gpu)
  .venv/bin/python scripts/bench_api.py --gpu --image onnx-lfm-api-gpu --configs q4,fp16
  # Thinking model (own models dir: same filenames as Instruct)
  .venv/bin/python scripts/bench_api.py --model-repo LiquidAI/LFM2.5-1.2B-Thinking-ONNX \\
      --models-dir ~/Development/onnx-lfm-api/models/lfm2.5-1.2b-thinking --max-tokens 2048 --configs q4:6
  # an already-running API
  .venv/bin/python scripts/bench_api.py --url http://jetson:8383/v1 --api-key ...
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onnx_lfm_agent.toolsets import build_registry  # noqa: E402

NAME, PORT = "lfm-bench", 8390


def workloads(max_tokens: int) -> dict:
    return {
        "generate": {"messages": [{"role": "user", "content":
            "Write a detailed paragraph of about 150 words on the history of the bicycle."}],
            "max_tokens": min(128, max_tokens), "temperature": 0},
        "agent": {"messages": [{"role": "user", "content":
            "The logs show upstream timeouts at 5s. Propose changing "
            "server.request_timeout_s in app.yaml to 15."}],
            "tools": build_registry("workspace", workspace="/nonexistent").schemas(),
            "max_tokens": max_tokens, "temperature": 0},
    }


def sh(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def cpu_usec() -> int | None:
    proc = sh("docker", "exec", NAME, "cat", "/sys/fs/cgroup/cpu.stat", check=False)
    for line in proc.stdout.splitlines():
        if line.startswith("usage_usec"):
            return int(line.split()[1])
    return None


def start_container(args, quant: str, threads: int) -> None:
    sh("docker", "rm", "-f", NAME, check=False)
    cmd = ["docker", "run", "-d", "--name", NAME, "-p", f"127.0.0.1:{PORT}:8383",
           "-v", f"{Path(args.models_dir).expanduser().resolve()}:/models:ro",
           "-e", f"LFM_QUANT={quant}", "-e", f"LFM_INTRA_OP_THREADS={threads}",
           "-e", "HF_HUB_OFFLINE=1", "-e", "LFM_API_KEY=",
           "-e", f"LFM_MAX_TOKENS={args.max_tokens}"]
    if args.model_repo:
        cmd += ["-e", f"LFM_MODEL_REPO={args.model_repo}"]
    if args.gpu:
        cmd += ["--gpus", "all"]
    sh(*cmd, args.image)


def wait_healthy(base: str, headers: dict, container: bool) -> dict:
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            r = httpx.get(base.removesuffix("/v1") + "/health", headers=headers, timeout=3)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        if container and sh("docker", "inspect", "-f", "{{.State.Running}}", NAME,
                               check=False).stdout.strip() != "true":
            raise RuntimeError("container exited:\n" + sh("docker", "logs", "--tail", "30", NAME,
                                                          check=False).stderr)
        time.sleep(1)
    raise RuntimeError("API did not become healthy")


def request(base: str, headers: dict, body: dict, containers: bool) -> dict:
    c0 = cpu_usec() if containers else None
    t0 = time.perf_counter()
    r = httpx.post(f"{base}/chat/completions", json={"model": "x", **body},
                   headers=headers, timeout=1800)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    usage, choice = r.json()["usage"], r.json()["choices"][0]
    c1 = cpu_usec() if containers else None
    return {"wall": wall, "prompt": usage["prompt_tokens"], "out": usage["completion_tokens"],
            "finish": choice["finish_reason"],
            "cores": (c1 - c0) / 1e6 / wall if c0 is not None and c1 is not None else None}


def measure(base: str, headers: dict, reps: int, max_tokens: int, containers: bool) -> dict:
    loads = workloads(max_tokens)
    request(base, headers, loads["generate"], containers)  # warm-up
    row = {}
    for name, body in loads.items():
        runs = [request(base, headers, body, containers) for _ in range(reps)]
        cores = [r["cores"] for r in runs if r["cores"] is not None]
        row[name] = {
            "wall_s": round(statistics.median(r["wall"] for r in runs), 2),
            "out_tok_s": round(statistics.median(r["out"] / r["wall"] for r in runs), 1),
            "tokens": f'{runs[0]["prompt"]}+{runs[0]["out"]}',
            "finish": runs[0]["finish"],
            "cores": round(statistics.median(cores), 1) if cores else None,
        }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="q4:6",
                    help="comma-separated quant[:threads], e.g. q4:0,q4:6,fp16 (threads 0 = ORT default)")
    ap.add_argument("--image", default="onnx-lfm-api:latest")
    ap.add_argument("--models-dir", default="~/Development/onnx-lfm-api/models")
    ap.add_argument("--model-repo", help="LFM_MODEL_REPO (default: the API's, i.e. Instruct)")
    ap.add_argument("--gpu", action="store_true", help="run containers with --gpus all")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--url", help="benchmark this running API instead of starting containers")
    ap.add_argument("--api-key", default="", help="for --url")
    ap.add_argument("--json", type=Path, help="also write results here")
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    results = []
    if args.url:
        health = wait_healthy(args.url.rstrip("/"), headers, container=False)
        targets = [(f"{health.get('quant')} @ {args.url}", None, None)]
    else:
        targets = []
        for spec in args.configs.split(","):
            quant, _, threads = spec.strip().partition(":")
            targets.append((f"{quant} threads={threads or 0}", quant, int(threads or 0)))

    try:
        for label, quant, threads in targets:
            base = args.url.rstrip("/") if args.url else f"http://127.0.0.1:{PORT}/v1"
            if not args.url:
                start_container(args, quant, threads)
                health = wait_healthy(base, headers, container=True)
            row = {"config": label, "providers": health.get("providers"),
                   **measure(base, headers, args.reps, args.max_tokens, not args.url)}
            results.append(row)
            g, a = row["generate"], row["agent"]
            print(f"{label:24} gen {g['out_tok_s']:6} tok/s {g['wall_s']:6}s | "
                  f"agent {a['wall_s']:6}s ({a['tokens']} tok, {a['finish']}) | "
                  f"cores {a['cores']} | {','.join(row['providers'] or [])}", flush=True)
    finally:
        if not args.url:
            sh("docker", "rm", "-f", NAME, check=False)
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
