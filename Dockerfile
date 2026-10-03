FROM python:3.13-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first (cached layer), pinned by uv.lock; then the package itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH \
    LFM_WORKSPACE=/workspace

# git: config changes are committed (and rolled back) in the workspace repo.
# The workspace is a bind mount owned by the host user, so mark it safe.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /workspace

# Tools execute inside this container, so run them unprivileged. Anything a
# tool may touch on the host must be mounted in deliberately (see compose).
RUN useradd --create-home --uid 1000 agent
COPY --from=build --chown=agent:agent /app/.venv /app/.venv
USER agent
WORKDIR /home/agent

# HTTP service port (lfm-agent-server; compose overrides the entrypoint).
EXPOSE 8384

# No args -> interactive REPL (needs `-it`); args -> one-shot prompt.
ENTRYPOINT ["lfm-agent"]
