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
    PATH=/app/.venv/bin:$PATH

# Tools execute inside this container, so run them unprivileged. Anything a
# tool may touch on the host must be mounted in deliberately.
RUN useradd --create-home --uid 1000 agent
COPY --from=build --chown=agent:agent /app/.venv /app/.venv
USER agent
WORKDIR /home/agent

# HTTP service port (lfm-agent-server; compose overrides the entrypoint).
EXPOSE 8384

# No args -> interactive REPL (needs `-it`); args -> one-shot prompt.
ENTRYPOINT ["lfm-agent"]
