# onnx-lfm-agent

A small **tool-calling agent harness** that drives [`onnx-lfm-api`](https://github.com/SonapSav/onnx-lfm-api)
over its OpenAI-compatible `/v1` endpoint. It is a **client** — it talks to the
API over HTTP and never imports the service.

## Status
Scaffold / work-in-progress. A working minimal agent loop + tool registry are
in place, with JSON-Schema argument validation; guardrails are next (see `CLAUDE.md`).

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
cp .env.example .env                                   # set LFM_API_KEY to the API's key
docker compose build
docker compose run --rm onnx-lfm-agent                 # interactive REPL
docker compose run --rm onnx-lfm-agent "what is 2+2?"  # one-shot
```

Dependencies are pinned in `uv.lock` (`uv lock` to refresh after editing
`pyproject.toml`). Tools run inside the container as an unprivileged user;
mount anything they need deliberately.

## Layout
- `config.py` — `LFM_*` settings (URL, api key, model, max rounds).
- `client.py` — OpenAI client pointed at the API.
- `tools.py` — `Tool` + `Registry` (schemas + dispatch; `dangerous` flag).
- `example_tools.py` — safe demo tools (time, add).
- `agent.py` — the model-decides / we-execute loop (`Agent`).
- `cli.py` — REPL / one-shot entrypoint (`lfm-agent`).

## Tests
```bash
pytest        # unit tests (no server needed)
```
