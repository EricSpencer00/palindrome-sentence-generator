from fastapi.testclient import TestClient

from server.mock import app


def test_standalone_mock_cannot_stream_legacy_palindrome_output():
    client = TestClient(app)
    response = client.get("/api/generate")
    assert response.status_code == 503
    assert "retired" in response.json()["detail"].lower()
    assert client.get("/health").json()["output_available"] is False
