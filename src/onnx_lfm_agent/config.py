from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Where the agent talks to the model. Overridable via LFM_* env / ./.env
    (same convention as the onnx-lfm-api service)."""

    model_config = SettingsConfigDict(env_prefix="LFM_", env_file=".env", extra="ignore")

    url: str = "http://127.0.0.1:8383/v1"  # LFM_URL — the onnx-lfm-api /v1 endpoint
    api_key: str = ""                       # LFM_API_KEY — only if the API has auth
    model: str = "lfm2.5"                   # name is cosmetic; the served model is fixed
    max_rounds: int = 6                     # safety cap on tool-call iterations
    temperature: float = 0.1                # mirrors the API default (Liquid-recommended)

    # Tools offered to the model: comma-separated toolsets (LFM_TOOLSETS).
    toolsets: str = "workspace"             # "workspace" (files + config), "demo" (time, add)
    workspace: str = "workspace"            # LFM_WORKSPACE — the only dir file tools may touch

    # LFM_SYSTEM_PROMPT: unset -> built-in (prompts.py); text -> replaces it; "" -> no system prompt.
    system_prompt: str | None = None

    # Per-tool policy overrides: "name=allow|ask|deny,..." (LFM_TOOL_POLICY).
    # Defaults: read-only tools allow, dangerous tools ask.
    tool_policy: str = ""

    # --- HTTP service (lfm-agent-server) ---
    agent_api_key: str = ""                 # LFM_AGENT_API_KEY — required; clients use it
    host: str = "127.0.0.1"                 # LFM_HOST — compose sets 0.0.0.0 in-container
    port: int = 8384                        # LFM_PORT


settings = Settings()
