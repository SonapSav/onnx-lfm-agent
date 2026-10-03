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

## Tool policy
Each tool is `allow` (runs), `ask` (runs only if approved) or `deny` (never
runs). Defaults: read-only tools `allow`, `dangerous=True` tools `ask`. The CLI
prompts for `ask`; the HTTP service has nobody to ask, so `ask` = denied there.
Override per tool with `LFM_TOOL_POLICY="name=allow,other=deny"`.

## Layout
- `config.py` — `LFM_*` settings (URL, api key, model, max rounds).
- `client.py` — OpenAI client pointed at the API.
- `tools.py` — `Tool` + `Registry` (schemas, dispatch, arg validation, allow/ask/deny policy).
- `example_tools.py` — safe demo tools (time, add).
- `agent.py` — the model-decides / we-execute loop (`Agent`).
- `cli.py` — REPL / one-shot entrypoint (`lfm-agent`).
- `server.py` — HTTP service (`lfm-agent-server`): `/health`, `POST /run`.

## Tests
```bash
pytest        # unit tests (no server needed)
```
