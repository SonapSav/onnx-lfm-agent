# onnx-lfm-agent

A small **tool-calling agent harness** that drives [`onnx-lfm-api`](https://github.com/SonapSav/onnx-lfm-api)
over its OpenAI-compatible `/v1` endpoint. It is a **client** — it talks to the
API over HTTP and never imports the service.

## Status
Scaffold / work-in-progress. A working minimal agent loop + tool registry are
in place, with JSON-Schema argument validation, a per-tool allow/ask/deny policy and an
HTTP service (see `CLAUDE.md`).

## Setup
Requires a running `onnx-lfm-api` server (e.g. `docker compose up -d` in that
repo; default `http://127.0.0.1:8383/v1`).

```bash
uv venv .venv --python 3.13          # system lacks python3.13-venv; use uv
uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env                 # set LFM_URL / LFM_API_KEY if the API has auth
```

## Run
```bash
.venv/bin/lfm-agent                              # interactive REPL
.venv/bin/lfm-agent "what time is it?"           # one-shot
```

## Docker
The API's compose stack must be up first — the agent joins its
`onnx-lfm-api_default` network and reaches it as `http://onnx-lfm-api:8383/v1`.

```bash
cp .env.example .env          # set LFM_API_KEY (the API's key) and LFM_AGENT_API_KEY
docker compose up -d --build  # HTTP service on 127.0.0.1:8384
docker compose run --rm onnx-lfm-agent                 # interactive REPL
docker compose run --rm onnx-lfm-agent "what is 2+2?"  # one-shot
```

Dependencies are pinned in `uv.lock` (`uv lock` to refresh after editing
`pyproject.toml`). Tools run inside the container as an unprivileged user;
mount anything they need deliberately.

## HTTP service
`lfm-agent-server` (the compose default service) exposes the agent loop:

```bash
curl -s http://127.0.0.1:8384/health                  # no auth
curl -s -X POST http://127.0.0.1:8384/run \
  -H "X-API-Key: $LFM_AGENT_API_KEY" -H 'content-type: application/json' \
  -d '{"prompt": "add 17.5 and 4.25"}'
# -> {"answer": "...", "steps": [{"tool","args","status","result"}], "history": [...]}
```

Pass `history` from a response back in to continue a conversation. The server
refuses to start without `LFM_AGENT_API_KEY` and is published on host loopback
only, because it executes tools.

## Tools
Default toolset `workspace` (`LFM_TOOLSETS`; `demo` adds `get_current_time`/`add`).
All file access is sandboxed to `LFM_WORKSPACE` — `..`, absolute paths and
symlinks leading out are refused, and `.git` is hidden.

| Tool | Policy | Does |
|---|---|---|
| `list_files` | allow | recursive listing (capped) |
| `read_file` | allow | numbered lines, line ranges, size-capped |
| `search_files` | allow | case-insensitive text search, `file:line: text` |
| `propose_config_change` | allow | edit **one** dotted key in a JSON/YAML file *in memory*; returns diff, schema validation (`<name>.schema.json` if present) and a `proposal_id`. Writes nothing. |
| `propose_fix_from_logs` | allow | harness-driven: code pulls the error lines from a log and lists the config's real keys (values, schema limits, comments); the model only picks the key, then the value (each a constrained tool call, retried on error); then `propose_config_change`. Writes nothing. Paths default to the only `.log` / config file. |
| `apply_config_change` | **ask** | write a valid proposal and `git commit` that file only (author `onnx-lfm-agent`, trailer `Agent-Proposal: pN`). Refuses if the file changed since the proposal. |

The CLI approval prompt shows the diff being applied. Undo is **operator-only**:

```bash
lfm-agent --rollback        # reverts the latest commit, only if the agent made it (-y: no prompt)
```

## Demo workspace
```bash
cp -r examples/workspace workspace
git -C workspace init -q -b main && git -C workspace add -A && git -C workspace commit -qm seed
lfm-agent "The logs show upstream timeouts at 5s. Propose changing server.request_timeout_s in app.yaml to 15, then apply it."
```
Create it before `docker compose up` (otherwise Docker creates it root-owned).

## System prompt
Off by default. `LFM_SYSTEM_PROMPT=builtin` sends a short built-in prompt (two
lines, `prompts.py`); any other text is sent as-is. It goes first on every model
call and is never stored in conversation history. Measured with the 1.2B model
and the current six tools (12 runs per scenario): without a prompt it passed
46/48 live-eval runs, with the built-in one 42/48. With the prompt it more often
applies a change you only asked it to propose, and sometimes refuses general
questions ("I can only list files").

## Tool policy
Each tool is `allow` (runs), `ask` (runs only if approved) or `deny` (never
runs). Defaults: read-only tools `allow`, `dangerous=True` tools `ask`. The CLI
prompts for `ask`; the HTTP service has nobody to ask, so `ask` = denied there.
Override per tool with `LFM_TOOL_POLICY="name=allow,other=deny"`.

## Layout
- `config.py` — `LFM_*` settings (URL, api key, model, max rounds).
- `client.py` — OpenAI client pointed at the API.
- `tools.py` — `Tool` + `Registry` (schemas, dispatch, arg validation, allow/ask/deny policy).
- `prompts.py` — the system prompt (base + toolset guidance).
- `toolsets.py` — builds the registry from `LFM_TOOLSETS`.
- `workspace.py` — the sandbox (path resolution) + git helper.
- `workspace_tools.py` — file tools, propose/apply, operator rollback.
- `log_fix.py` — the `propose_fix_from_logs` workflow (evidence, key list, two narrow model decisions)
  and the setting resolver `propose_config_change` uses for unknown keys / ambiguous files.
- `evals.py` — live eval scenarios (A–H, plus JSON scenario files via `load_scenarios`, e.g. the held-out
  `examples/workspace-heldout/scenarios.json`) shared by `scripts/eval_live.py` and the integration tests.
- `config_edit.py` — one-key JSON/YAML edits (comment/indent-preserving), schema validation, diff.
- `example_tools.py` — demo tools (time, add), toolset `demo`.
- `agent.py` — the model-decides / we-execute loop (`Agent`).
- `cli.py` — REPL / one-shot entrypoint (`lfm-agent`).
- `server.py` — HTTP service (`lfm-agent-server`): `/health`, `POST /run`.

## Tests
```bash
pytest                    # unit tests: offline, fast (integration tests deselected)
pytest -m integration     # live: needs a running onnx-lfm-api at LFM_URL (~2 min)
```

## Evaluating models and hardware
Both scripts read `LFM_URL` / `LFM_API_KEY` / `LFM_TEMPERATURE` from the
environment or `./.env`, so they work against any API (laptop, GPU PC, Jetson).

```bash
# Agent behaviour with the real model: scenarios A-D (see the script's --help),
# each run in a fresh temp git workspace seeded from examples/workspace.
.venv/bin/python scripts/eval_live.py                    # 6 runs each, system prompt on
.venv/bin/python scripts/eval_live.py --compare-prompt   # prompt off vs on
LFM_URL=http://gpu-pc:8383/v1 .venv/bin/python scripts/eval_live.py -n 3

# API speed: throwaway containers per config (production untouched) ...
.venv/bin/python scripts/bench_api.py --configs q4:0,q4:6,q4f16:6
.venv/bin/python scripts/bench_api.py --gpu --image onnx-lfm-api-gpu --configs q4,fp16
# ... or an API that is already running
.venv/bin/python scripts/bench_api.py --url http://jetson:8383/v1 --api-key ...
```
