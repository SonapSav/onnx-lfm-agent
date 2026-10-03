from __future__ import annotations

from openai import OpenAI

from .config import settings


def make_client() -> OpenAI:
    """OpenAI client pointed at the onnx-lfm-api /v1 endpoint."""
    return OpenAI(base_url=settings.url, api_key=settings.api_key or "no-auth")
