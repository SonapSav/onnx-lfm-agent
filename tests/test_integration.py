"""Live checks against a running onnx-lfm-api (real model).

Run with:  pytest -m integration        (uses LFM_URL / LFM_API_KEY / ./.env)
Skipped when the API isn't reachable. Only scenarios that pass reliably with
LFM2.5-1.2B-Instruct are asserted (>= 11/12 in live evals); the rest are measured
by scripts/eval_live.py instead.
"""

import httpx
import pytest

from onnx_lfm_agent.config import settings
from onnx_lfm_agent.evals import SCENARIOS, run_scenario

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def api_up():
    try:
        httpx.get(settings.url.rstrip("/").removesuffix("/v1") + "/health", timeout=3).raise_for_status()
    except httpx.HTTPError as e:
        pytest.skip(f"onnx-lfm-api not reachable at {settings.url}: {e}")


@pytest.mark.parametrize("key", ["A", "C", "D", "E", "F", "G", "H"])
def test_reliable_scenarios(key):
    outcome = run_scenario(SCENARIOS[key])
    assert outcome.passed, (f"{key} failed: tools={outcome.tools} "
                            f"answer={outcome.result.answer[:200]!r}")


@pytest.mark.parametrize("key", ["B", "C"])
def test_config_stays_valid_even_when_the_model_fails(key):
    """B still fails sometimes with this model; the config must survive regardless."""
    outcome = run_scenario(SCENARIOS[key])
    assert outcome.config_ok, f"app.yaml invalid after {key}: tools={outcome.tools}"
