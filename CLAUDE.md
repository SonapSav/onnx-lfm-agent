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

## What's already here (the scaffold)
- `tools.py` — `Tool` dataclass + `Registry` (`@registry.tool(...)`, `.schemas()`, `.get()`), with a `dangerous` flag for side-effecting tools.
  `Tool.validate(args)` checks args against `parameters` (jsonschema, Draft 2020-12); schemas are `check_schema`'d at registration.
- `agent.py` — `Agent.run()`: the model-decides → execute → feed-result-back loop, capped by `max_rounds`; `dangerous` tools gate on an `approve` callback.
  `_execute` validates args **before** approval; invalid → function not called, model gets `{error, details, expected: <schema>}` and can retry.
- `example_tools.py` — safe demo tools (time, add).
- `cli.py` — `lfm-agent` REPL / one-shot, with an interactive approver.
- `tests/test_registry.py`, `tests/test_validation.py` — unit tests (no server; `FakeClient` replays scripted model turns).

Verified: unit test passes; the loop runs against a live API (model picks tools,
client executes, results feed back).

## Roadmap — what to build next (rough priority)
1. ~~**Arg validation**~~ — DONE (reject-only). Possible follow-up: narrow opt-in repair (numeric strings → numbers, drop undeclared keys) *only if* live runs show the model making those mistakes. Note: example schemas don't set `additionalProperties: false`, so extra args pass validation and surface as a `TypeError` from the call.
2. **Guardrails / safety**: richer `approve` policy (per-tool, confidence/risk based); for the log→config class of tasks, enforce **propose → validate → apply** (dry-run/diff, schema check, git commit + rollback) rather than direct side effects.
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

## Git
- Remote `origin` → https://github.com/SonapSav/onnx-lfm-agent (**public**); `main` tracks `origin/main`.
- Repo-local identity Panos Vasilopoulos <sonap.sav@gmail.com> (matches the API repo).
  The global identity on this host is different (`dev@primesoft.ae`) — don't rely on it.
- Repo-local credentials: for `https://github.com` the global `store` helper is
  cleared and `!gh auth git-credential` is used with username `SonapSav`. This
  follows gh's *active* account (several are logged in), so if pushes fail or
  land under the wrong account: `gh auth switch -u SonapSav`.
