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
- `toolsets.py` — `build_registry(LFM_TOOLSETS)`: `workspace` (default) and/or `demo` (`example_tools.py`: time, add).
- `workspace.py` — `Workspace.resolve()` sandbox (relative to `LFM_WORKSPACE`; `..`/absolute/symlink escapes and `.git`
  refused) + `git()` helper (author `onnx-lfm-agent`) + a lock for git-mutating ops.
- `workspace_tools.py` — `list_files`, `read_file`, `search_files` (allow, capped); `propose_config_change` (allow,
  writes nothing → diff + validation + `proposal_id`; refuses dirty/untracked files; in-memory `ProposalStore`, cap 50);
  `apply_config_change` (dangerous/ask; sha256 staleness check, commits only that file with `Agent-Proposal: pN`
  trailer, restores the file if the commit fails; `preview` shows the diff to the approver).
  **Rollback is operator-only**: `rollback_last_change()` / `lfm-agent --rollback [-y]`, reverts HEAD only if it has the trailer.
- `config_edit.py` — one dotted key per edit (ruamel round-trip keeps YAML comments/indent; JSON keeps indent),
  schema = sibling `<stem>.schema.json`, unified diff; errors list existing keys so the model can retry.
- `examples/workspace/` — demo `app.yaml` + schema + `logs/app.log` (timeouts); copy to `./workspace` (gitignored, own git repo).
- `cli.py` — `lfm-agent` REPL / one-shot, interactive approver for `ask` tools (shows `preview`; EOF → deny), `--rollback`.
- `tests/` — unit tests, no server: `fakes.py` (`FakeClient` replays scripted model turns, `reply`, `tool_call`),
  `test_registry`, `test_validation`, `test_policy`, `test_server` (FastAPI `TestClient`), `test_workspace_tools`
  (temp git repo seeded from `examples/workspace`). Tests pass `tool_policy=""`
  so a local `LFM_TOOL_POLICY` can't leak in.

Verified: 91 unit tests pass; live: CLI (venv + compose container) and the HTTP service against the running API,
including propose → approve → commit → operator rollback on the demo workspace.

## Roadmap — what to build next (rough priority)
1. ~~**Arg validation**~~ — DONE (reject-only at the agent level). Narrow repairs live *inside* `propose_config_change`
   (see gotchas), added after live runs showed the mistakes. Example schemas don't set `additionalProperties: false`, so extra
   args pass validation and surface as a `TypeError` from the call.
2. ~~**Guardrails / safety**~~ — DONE: per-tool allow/ask/deny (headless = deny) and **propose → validate → apply**
   (diff, schema check, git commit, operator rollback). Maybe later: `POST /proposals/{id}/apply` so a human can approve
   server-side proposals over HTTP; risk/confidence-based approval.
3. **Real tools** — first batch DONE (workspace files + config propose/apply). Next candidates: HTTP GET (host allowlist),
   DB read-only query. Keep the set small (now 5 tools) — this is a 1.2B model.
   **Known limit:** open-ended "read the logs and fix the config" fails (0/6 live): the model guesses key names
   (`server.timeout`) instead of reading `app.yaml`, and doesn't use the key list in the error. Directed requests
   ("change server.request_timeout_s to 15, then apply") succeed 6/6. Likely fixes: a system prompt that says
   "read the config before proposing" (item 6), or offering fewer tools per step.
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
- **Live-observed model mistakes (LFM2.5-1.2B) and how they're handled** — all in `propose_config_change`, each
  reported to the model in `note`:
  - quotes numbers/booleans (`"15"`) → converted only if the schema rejects the string and accepts the conversion,
    or (no schema) the current value is that kind. Typing `value` in the tool schema instead backfired
    (number-first type list made it send `0` for `"debug"`), so `value` stays untyped.
  - omits `path` / folds the file name into the key (`app.yaml.logging.level`) → `path` is optional: taken from the
    key prefix, else the only config file in the workspace, else refused with the candidates. A clearer param
    description made it *worse*.
  - called `rollback_config` when asked to apply → rollback removed from the model's tools (operator-only).
  Measured: directed propose+apply went 2/8 → 12/12 with these.
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
