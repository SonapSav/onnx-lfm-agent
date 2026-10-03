"""HTTP service: POST /run drives the agent loop for non-interactive callers.

There is no approver here, so tools with policy "ask" are always denied; use
LFM_TOOL_POLICY to allow specific tools explicitly.
"""

from __future__ import annotations

import hmac
import logging
import sys
from typing import Any

import openai
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from .agent import Agent
from .config import settings
from .example_tools import registry

log = logging.getLogger(__name__)


class RunRequest(BaseModel):
    prompt: str
    history: list[dict[str, Any]] = []  # `history` from a previous response


class RunResponse(BaseModel):
    answer: str
    steps: list[dict[str, Any]]
    history: list[dict[str, Any]]


def create_app(agent: Agent | None = None, api_key: str | None = None) -> FastAPI:
    """Build the app. Refuses to run without a key: this endpoint executes tools."""
    key = settings.agent_api_key if api_key is None else api_key
    if not key:
        raise RuntimeError("LFM_AGENT_API_KEY is not set; refusing to start the agent "
                           "server without auth (it executes tools)")
    agent = agent or Agent(registry)  # no approver: "ask" tools are denied

    def require_api_key(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> None:
        """``X-API-Key: <key>`` or ``Authorization: Bearer <key>``, constant-time
        compare (same scheme as onnx-lfm-api)."""
        provided = x_api_key
        if not provided and authorization and authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        if not provided or not hmac.compare_digest(provided, key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    app = FastAPI(title="onnx-lfm-agent", version="0.0.1")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "tools": agent.policies}

    # Plain `def`: FastAPI runs it in a worker thread, so a long agent run
    # doesn't block /health.
    @app.post("/run", response_model=RunResponse, dependencies=[Depends(require_api_key)])
    def run(req: RunRequest) -> RunResponse:
        try:
            result = agent.run(req.prompt, req.history)
        except openai.APIError as e:
            log.error("model API call failed: %s", e)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                                detail=f"model API error: {e}") from e
        return RunResponse(answer=result.answer,
                           steps=[s.to_dict() for s in result.steps],
                           history=result.messages)

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not settings.agent_api_key:
        sys.exit("LFM_AGENT_API_KEY is not set; refusing to start the agent server "
                 "without auth (it executes tools).")
    # Single worker: one shared agent; the model API serializes generation anyway.
    uvicorn.run("onnx_lfm_agent.server:create_app", factory=True,
                host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
