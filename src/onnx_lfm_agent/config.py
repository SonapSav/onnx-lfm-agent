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
    temperature: float = 0.0


settings = Settings()
