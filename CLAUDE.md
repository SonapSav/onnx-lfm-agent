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
- `prompts.py` — system prompt = 2-line `BASE` + `Registry.guidance` lines from toolsets (workspace contributes
  **none**, on purpose — see gotchas). `Agent` sends it as the leading system message on **every** call but never
  stores it in returned history (a leading system message in incoming history is dropped). `LFM_SYSTEM_PROMPT`:
  unset/`""` → off (**default since propose_fix_from_logs**, user decision 2026-10-03), `builtin` → `prompts.py`,
  other text → used as-is. The built-in one costs +34 prompt tokens.
- `toolsets.py` — `build_registry(LFM_TOOLSETS)`: `workspace` (default) and/or `demo` (`example_tools.py`: time, add).
- `workspace.py` — `Workspace.resolve()` sandbox (relative to `LFM_WORKSPACE`; `..`/absolute/symlink escapes and `.git`
  refused) + `git()` helper (author `onnx-lfm-agent`) + a lock for git-mutating ops.
- `workspace_tools.py` — `list_files`, `read_file`, `search_files` (allow, capped); `propose_config_change` (allow,
  writes nothing → diff + validation + `proposal_id`; refuses dirty/untracked files; in-memory `ProposalStore`, cap 50);
  `apply_config_change` (dangerous/ask; sha256 staleness check, commits only that file with `Agent-Proposal: pN`
  trailer, restores the file if the commit fails; `preview` shows the diff to the approver).
  **Rollback is operator-only**: `rollback_last_change()` / `lfm-agent --rollback [-y]`, reverts HEAD only if it has the trailer.
  `propose_fix_from_logs` (allow, `log_fix.py`): **harness-driven workflow** for open-ended "check the logs and fix it" —
  code extracts error lines (dedup, counts) and lists the config's leaf keys (value, schema limits, YAML comment); the
  model makes two narrow single-tool decisions, `choose_setting(key: enum of real keys)` then `set_value(value: the
  key's own schema)`, retried with errors fed back (unchanged value refused); then `propose_config_change`. Result has
  `summary` + `next` ("report; apply only if asked") — without it the outer model re-proposed garbage and applied that.
  Its own model calls use the registry's `client` (`build_registry(client=...)`, default LFM_URL). **Run-scoped
  guard:** `Agent.run()` sets `tools.CURRENT_RUN`; proposals record it; the tool refuses (tool error naming the
  pending `pN`) if this run already has an unapplied proposal for the same file. Without it, directed A ("set X to
  15, then apply") sometimes also called the tool, got 10 as `p2` and applied that (A 10/12). A "use only when the
  user did not name the setting" description didn't help (still fix calls); the tool error did (A 12/12).
- `config_edit.py` — one dotted key per edit (ruamel round-trip keeps YAML comments/indent; JSON keeps indent),
  schema = sibling `<stem>.schema.json`, unified diff; errors list existing keys so the model can retry.
- `examples/workspace/` — demo `app.yaml` + schema + `logs/app.log` (timeouts); copy to `./workspace` (gitignored, own git repo).
- `cli.py` — `lfm-agent` REPL / one-shot, interactive approver for `ask` tools (shows `preview`; EOF → deny), `--rollback`.
- `evals.py` + `scripts/eval_live.py` — live eval scenarios A–D (directed apply / directed propose / open-ended /
  no-tools) against the real model; fresh temp git workspace per run (seeded from `examples/workspace`, so it never
  touches `./workspace` and works on any machine); prompt off/on comparison; reports pass rate, s/call, out tokens,
  truncations, tool paths. `scripts/bench_api.py` — API speed (gen tok/s + tool-calling request): throwaway
  containers per `quant[:threads]` config (`--gpu`, `--image`, `--model-repo`, `--models-dir`), or `--url` for a
  running API (Jetson). These replace the ad-hoc scratch scripts used for every number in this file.
- `tests/` — unit tests, no server: `fakes.py` (`FakeClient` replays scripted model turns, `reply`, `tool_call`),
  `test_registry`, `test_validation`, `test_policy`, `test_server` (FastAPI `TestClient`), `test_workspace_tools`
  (temp git repo seeded from `examples/workspace`). Tests pass `tool_policy=""`
  so a local `LFM_TOOL_POLICY` can't leak in.
  `test_integration.py` (`-m integration`, deselected by default via `addopts`; skips if the API is down): asserts
  only what Instruct does reliably (A, D) plus the invariant that `app.yaml` stays schema-valid after B/C runs.

Verified: 109 unit + 5 integration tests pass (live evals 48/48 with the default, prompt off); live: CLI (venv + compose container) and the HTTP service against the running API,
including propose → approve → commit → operator rollback on the demo workspace.

## Roadmap — what to build next (rough priority)
1. ~~**Arg validation**~~ — DONE (reject-only at the agent level). Narrow repairs live *inside* `propose_config_change`
   (see gotchas), added after live runs showed the mistakes. Example schemas don't set `additionalProperties: false`, so extra
   args pass validation and surface as a `TypeError` from the call.
2. ~~**Guardrails / safety**~~ — DONE: per-tool allow/ask/deny (headless = deny) and **propose → validate → apply**
   (diff, schema check, git commit, operator rollback). Maybe later: `POST /proposals/{id}/apply` so a human can approve
   server-side proposals over HTTP; risk/confidence-based approval.
3. **Real tools** — first batch DONE (workspace files + config propose/apply). Next candidates: HTTP GET (host allowlist),
   DB read-only query. Keep the set small (now 6 tools) — this is a 1.2B model.
   Open-ended "read the logs and fix the config" was 0/6 under every prompting variant (Instruct guesses key names
   and never uses what it reads). **Solved by the harness-driven `propose_fix_from_logs`** (see above): C 12/12 with
   the system prompt off and on. Probing showed key and value must be separate decisions: asked together it picked
   the right key 8/8 but copied the current value back 8/8; asked alone it raised the value 6/6. Saying "must differ
   from 5" made it worse (chose 3) — code enforces that instead. Pattern for future open-ended tasks: code plans,
   model fills enum-constrained single decisions.
4. **Streaming** of assistant text + tool-call deltas (API already streams; surface it).
5. **Tracing/logging** of each step (prompt, tool calls, results) for debugging and evals.
6. **System prompt** — DONE (built-in base prompt, now default **off**; `LFM_SYSTEM_PROMPT=builtin`). Still open: state/memory beyond the raw message list
   (e.g. trimming long histories). The API's prompt-prefix cache now skips re-reading the shared start of each
   round, but a longer history still means bigger snapshots and more new tokens per round.
7. ~~**Tests**~~ — DONE: `pytest -m integration` (live, ~2 min) + `scripts/eval_live.py` / `scripts/bench_api.py`.
8. **GPU / Jetson** (target decided: **deploy on Jetson Orin class, develop on an x86 PC with a CUDA GPU**; this
   laptop has no NVIDIA GPU). Agent side needs nothing: point `LFM_URL` at the GPU box's API.
   **x86 dev box up (2026-10-03):** `ssh gpu` (alias in `~/.ssh/config` on the laptop), GTX 1660 6 GB
   (Turing, no tensor cores), driver 615, Docker + nvidia runtime. Both repos cloned in `~/Development/`; the API runs
   from the GPU compose files plus an untracked `docker-compose.local.yml` (publishes on `127.0.0.1:8383` only, q4).
   Reach it with a tunnel: `ssh -f -N -L 18383:127.0.0.1:8383 gpu`, then `LFM_URL=http://127.0.0.1:18383/v1 LFM_API_KEY=`
   (no auth on that API). Findings: `Dockerfile.gpu` needed a **CUDA 13** base (the ORT 1.30 wheel is a CUDA 13 build);
   on the 1660 **q4 beats fp16** (fp16 math is slow without tensor cores; see the API README). Agent round 1.35 s
   vs 6.46 s on the laptop CPU, gen 123 vs 18 tok/s; live evals identical to CPU (prompt off A6 B6 C0 D0, on A6 B0
   C1 D6) at 1.5–2.2 s/call. **IO binding DONE** (API `LFM_IO_BINDING`, auto = CUDA + fp16/bf16 cache): q4f16 decode
   1.36x at 2.3k context, token-identical. It can't help q4 here: ORT's CUDA attention op is fp16/bf16-only, so with
   q4's fp32 cache attention runs **on the CPU**. **Tried and dropped: fp16 attention in the q4 graph** (fp16 cache,
   Casts around the 6 GQA nodes, so attention runs on CUDA). Greedy tokens identical to q4, but decode at 64/700/2000
   ctx was 145/99/60 tok/s with binding vs q4's 145/106/67, and prefill was unchanged. On Turing, ORT's CUDA GQA is no
   faster than its CPU one (no fused attention before sm_80), so the slowdown with context comes from the kernel,
   not the copies. Re-try on Ampere/Orin only. Still open: an
   aarch64/JetPack image for Orin, re-measure fp16 there. Dev flow: edit on the laptop, `rsync` the API tree to the
   server, rebuild, bench through the tunnel; commit once measured.

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
- **Performance / CPU**: all the load is the API's inference (the agent idles at ~0.3% CPU), so agent
  workers don't help. The API is tuned on this host to `LFM_QUANT=q4`, `LFM_INTRA_OP_THREADS=6`
  (6c/12t Ryzen; ORT's default of all 12 threads was ~1.7x slower per tool-calling round at double the
  CPU) — see the API repo's README "Performance tuning" / commit `9b493f7`. One agent round ≈ 6.5 s
  uncached, mostly prompt processing (tool schemas + history). **The API's prompt-prefix cache** (API README,
  `LFM_PREFIX_CACHE_SIZE`) reuses the system+tools state and the previous round's conversation: a repeated agent
  request takes 2.06 s on the laptop CPU and 0.36 s on the GTX 1660 (live-eval calls 0.5–0.9 s on GPU, 141/148 hits).
  Tool descriptions now cost once per snapshot instead of on every call, but they still cost on cache misses.
  **A second API instance does not help** (measured, batch of 6 agent calls: 1×6 threads 38.6 s; 2×3 threads
  concurrent 37.9 s, within noise; 2×6 threads 55.7 s). Inference is memory-bandwidth bound, so instances just
  split it. To speed up live testing: the GPU box (`ssh gpu`, see roadmap 8), shorter prompts, fewer runs.
- **Live evals: system prompt & guard** (LFM2.5-1.2B-Instruct q4, 6 runs each; A = "propose X, then apply",
  B = "propose X" (must not apply), C = open-ended "check logs and fix", D = "capital of France?" (no tools)):

  | variant | A | B | C | D |
  |---|---|---|---|---|
  | no system prompt | 6 | **6** | 0 | **0** (refuses: "functions are focused on file management") |
  | base prompt (shipped until propose_fix_from_logs) | 6 | 0 | 0 | **6** |
  | + workspace rules ("read the file first", "only apply if asked") | 6 | 0 | 0* | 5 |
  | + read-before-propose guard (removed) | 6 | 0–1 | 0 | 0 / 6 |

  \*first run scored 6/6 only because the example key in the prompt *was* the answer — never use a real key as an
  example. **Any** system prompt makes Instruct apply when only asked to propose (safe: CLI asks, server denies) —
  accepted in exchange for D. The guard derailed the model (extra round → `read_file(".")`, applying proposals that
  don't exist). Lesson: this model follows tool **errors** and narrow tool design, not instructions.
- **Live evals with `propose_fix_from_logs`** (GTX 1660, 12 runs each): prompt **off** A 11 B 12 C 12 D 11 (46/48);
  prompt **on** (default) A 12 B 9 C 12 D 9 (42/48). The 6th tool flipped the old trade-off: D no longer needs the
  prompt (it was 0/6 without), and the prompt now costs B and D. **Default switched to off** (user, 2026-10-03).
  With the run-scoped guard, prompt off: **A 12 B 12 C 12 D 12 (48/48)**, 0.5–0.7 s/call.
  Adding "Answer general questions from your own knowledge." to BASE made it worse (B 1/6, D 1/6 with prompt on) —
  reverted. Live check C also requires nothing but the timeout to have changed (a run that applied port=10 passed
  the old check).
- **LFM2.5-1.2B-Thinking evaluated, not adopted (for now).** ONNX build `LiquidAI/LFM2.5-1.2B-Thinking-ONNX`, same
  template/tool format. On this CPU: ~1000–1350 reasoning tokens per round → 68–87 s/call (Instruct ~8 s); at 1024
  max_tokens every call was cut off mid-thought. Quality was better (right key, included `path`, did *not* apply
  in B), still quoted `"15"` and invented a `proposal_id`. `<think>`/`</think>` are **not** special tokens
  (text survives decoding; the API doesn't split it out yet). Its files live in the API repo's
  `models/lfm2.5-1.2b-thinking/` (separate dir — same filenames as Instruct, would overwrite).
  **GTX 1660 re-eval (q4, max_tokens 2048, prompt on, 3 runs):** A 1/3 (2 runs cut off at 2048), B **3/3** (Instruct
  0/6), C 0/3 (but it now reads the files), D 3/3; 780–1450 out tokens, 10–21 s/call (Instruct 1.5–2.2 s). Still not
  adopted: only B improves, and it costs ~10x per call. Worth re-trying with 4096 max_tokens on a faster GPU.
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
