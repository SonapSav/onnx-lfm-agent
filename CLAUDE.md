# CLAUDE.md — onnx-lfm-agent

Handoff for building this out. Keep it current.

## What this is (and isn't)
A **tool-calling agent harness** that consumes the `onnx-lfm-api` service over
its OpenAI `/v1` endpoint. It is a **pure client**: HTTP only, **never import
the API package**. The two projects are deliberately separate — different
lifecycle, deps, and trust boundary (this one *executes tools*; the API only
*decides*).

- Sibling API repo: `~/Development/onnx-lfm-api` → GitHub `SonapSav/onnx-lfm-api`.
  Run it first: `docker compose up -d` there. Default endpoint
  `http://127.0.0.1:8383/v1`; auth key (if set) is in that repo's `.env`.
- The API already supports OpenAI tool calling (request `tools`, response
  `tool_calls`, `role:"tool"` results round-trip). This project builds *on top*.

## Setup (same host quirks as the API)
- Use **uv** for the venv — the system lacks `python3.13-venv`/ensurepip:
  ```bash
  uv venv .venv --python 3.13
  uv pip install --python .venv/bin/python -e ".[dev]"
  ```
- Point at the API via `LFM_URL` / `LFM_API_KEY` (`./.env`, `.env.example`).
- **Docker**: `Dockerfile` (2-stage, `python:3.13-slim` + uv, `uv sync --frozen`
  from `uv.lock`, runs as non-root `agent`, `ENTRYPOINT ["lfm-agent"]`, `EXPOSE 8384`).
  `docker-compose.yml` joins the API's external network `onnx-lfm-api_default`
  (start the API first) and defaults `LFM_URL=http://onnx-lfm-api:8383/v1`. Services:
  - `onnx-lfm-agent-server` — `docker compose up -d`; entrypoint `lfm-agent-server`,
    published on **`127.0.0.1:8384` only** (it executes tools; drop the prefix for LAN),
    healthcheck on `/health`, `restart: unless-stopped`.
  - `onnx-lfm-agent` — CLI under profile `cli` (not started by `up`):
    `docker compose run --rm onnx-lfm-agent [prompt]`.
- `uv.lock` pins deps; after editing `pyproject.toml` run `uv lock` and commit it.

## What's already here
- `tools.py` — `Tool` dataclass + `Registry` (`@registry.tool(...)`, `.schemas()`, `.get()`, iterable).
  `Tool.validate(args)` checks args against `parameters` (jsonschema, Draft 2020-12); schemas are `check_schema`'d at registration.
  `Tool.policy` = `allow | ask | deny`; default `ask` if `dangerous=True`, else `allow`. `parse_policy_spec()` parses `LFM_TOOL_POLICY`.
- `agent.py` — `Agent.run()` → `RunResult(answer, messages, steps)`. Loop: model decides → per call: unknown? → parse →
  **validate** → **policy** → execute; every outcome is a `Step` (status `executed | tool_error | invalid_args | denied |
  unknown_tool | bad_json`), logged, and fed back to the model. `messages` are plain dicts (JSON) and include the final answer.
  Policy: `allow` runs; `deny` refuses; `ask` calls `approve` and is **denied when there's no approver**.
  `LFM_TOOL_POLICY` overrides per tool; unknown names fail at `Agent()` init; loosening a dangerous tool to `allow` logs a warning.
- `server.py` — `lfm-agent-server` (FastAPI/uvicorn, 1 worker, `create_app()` factory). `GET /health` (no auth; lists tool
  policies), `POST /run {prompt, history?}` → `{answer, steps, history}`; model-API errors → 502. Auth = `LFM_AGENT_API_KEY`
  via `X-API-Key` or `Bearer` (constant-time, same scheme as the API); **refuses to start without it**. No approver → `ask` tools denied.
- `example_tools.py` — safe demo tools (time, add).
- `cli.py` — `lfm-agent` REPL / one-shot, with an interactive approver for `ask` tools (EOF → deny).
- `tests/` — unit tests, no server: `fakes.py` (`FakeClient` replays scripted model turns, `reply`, `tool_call`),
  `test_registry`, `test_validation`, `test_policy`, `test_server` (FastAPI `TestClient`). Tests pass `tool_policy=""`
  so a local `LFM_TOOL_POLICY` can't leak in.

Verified: 37 unit tests pass; live: CLI (venv + compose) and the HTTP service against the running API.

## Roadmap — what to build next (rough priority)
1. ~~**Arg validation**~~ — DONE (reject-only). Possible follow-up: narrow opt-in repair (numeric strings → numbers, drop undeclared keys) *only if* live runs show the model making those mistakes. Note: example schemas don't set `additionalProperties: false`, so extra args pass validation and surface as a `TypeError` from the call.
2. **Guardrails / safety** — per-tool allow/ask/deny policy DONE (incl. headless = deny). Still open:
   **propose → validate → apply** (dry-run/diff, schema check, git commit + rollback) for the log→config class of
   tasks — deliberately deferred to land *with the first real config-changing tool* (item 3), so it's designed
   against a concrete case. Maybe later: risk/confidence-based approval.
3. **Real tools** beyond the demos (HTTP, files, shell, DB) — keep the set small and descriptions crisp; this is a 1.2B model and degrades with large/ambiguous toolboxes.
4. **Streaming** of assistant text + tool-call deltas (API already streams; surface it).
5. **Tracing/logging** of each step (prompt, tool calls, results) for debugging and evals.
6. **State/memory** across turns beyond the raw message list; maybe a system prompt / persona.
7. **Tests**: an integration test behind a marker that needs a running API (mirror the API repo's `-m integration` pattern).

## Design notes / gotchas
- The model (LFM2.5-1.2B) can be **over-eager** — may call an unnecessary tool (seen: calling `get_current_time` before a weather lookup). Harness should tolerate/ignore irrelevant results; consider narrowing offered tools per step.
- Tool `arguments` arrive as a **JSON string** — `json.loads` before executing.
- **You own execution = you own safety.** Never blindly dispatch; gate anything side-effecting.
- `tool_choice` is accepted by the API but not enforced — the model decides.
- `run()` must append the final assistant answer to `messages` (it once didn't,
  and the REPL / `/run` history forgot the model's own replies) — covered by tests.
- **Temperature** defaults to **0.1** (= the API's Liquid-recommended default; the
  agent always sends it, so it overrides the server's value). Over `/v1` only
  temperature is client-settable — `top_k=50` / `repetition_penalty=1.05` are
  fixed server-side (penalty applies even at 0). 0.0 = exact greedy: use it for
  reproducible tests/evals, but it makes validation-error retries likelier to
  repeat the same bad call.

## Git
- Remote `origin` → https://github.com/SonapSav/onnx-lfm-agent (**public**); `main` tracks `origin/main`.
- Repo-local identity Panos Vasilopoulos <sonap.sav@gmail.com> (matches the API repo).
  The global identity on this host is different (`dev@primesoft.ae`) — don't rely on it.
- Repo-local credentials: for `https://github.com` the global `store` helper is
  cleared and `!gh auth git-credential` is used with username `SonapSav`. This
  follows gh's *active* account (several are logged in), so if pushes fail or
  land under the wrong account: `gh auth switch -u SonapSav`.
