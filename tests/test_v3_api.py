"""The legacy v3 API must fail closed rather than serve shortcut material."""
from fastapi.testclient import TestClient

from server import v3


def test_legacy_bank_has_no_shared_gate_survivor():
    assert v3._load_bank() == []


def test_api_refuses_to_serve_legacy_material():
    v3._bank = []
    v3._tables = None
    v3._load_error = None
    from server.app import app

    client = TestClient(app)
    response = client.get("/api/v3/palindrome")
    assert response.status_code == 503
    assert "blinded human-reader evidence" in response.json()["detail"]

    health = client.get("/api/v3/health").json()
    assert health["ok"] is False
    assert health["bank"] == 0


def test_every_legacy_output_route_is_retired():
    from server.app import app

    client = TestClient(app)
    for path in (
        "/api/generate", "/api/v2/paragraph", "/api/v2/generate",
        "/api/v3/composition", "/api/v3/palindrome", "/api/v3/refrain",
    ):
        response = client.get(path)
        assert response.status_code == 503, path
        assert "blinded human-reader evidence" in response.json()["detail"]
