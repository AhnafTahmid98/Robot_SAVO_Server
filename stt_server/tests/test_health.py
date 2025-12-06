"""
Robot Savo — STT + LLM gateway tests: /health endpoint.

These tests are intentionally small but high-signal:

- Verify that the /health endpoint is reachable (200 OK).
- Verify that the response schema matches HealthResponse.
- Ensure the service can be imported and started without side effects.

Run from the stt_server root:

    cd ~/Robot_Savo_Server/stt_server
    python -m pytest -q

Requires `pytest` and `httpx`/`requests` via `fastapi[all]` / `uvicorn[standard]`.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import HealthResponse


client = TestClient(app)


def test_health_status_code() -> None:
    """
    /health should be reachable and return HTTP 200.
    """
    response = client.get("/health")
    assert response.status_code == 200, f"/health returned {response.status_code}"


def test_health_response_shape() -> None:
    """
    /health should return a JSON object compatible with HealthResponse.

    This also ensures that the Pydantic model and FastAPI wiring are correct.
    """
    response = client.get("/health")
    data = response.json()

    # Validate with Pydantic model (raises if invalid)
    health = HealthResponse(**data)

    assert isinstance(health.status, str)
    assert health.status.lower() in {"ok", "healthy"}
