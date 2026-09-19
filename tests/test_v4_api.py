"""v4 exposes evidence and diagnostics while keeping generation gated."""
import hashlib

from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_v4_health_reports_evidence_gate():
    response = client.get("/api/v4/health")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "v4"
    assert body["gate"]["generation"] == "gated"
    assert body["gate"]["reader_evidence"] is False
    assert body["best_known_letters"] == 38


def test_v4_evidence_contains_actual_rendered_candidate_and_independent_audit():
    response = client.get("/api/v4/evidence")
    assert response.status_code == 200
    body = response.json()
    candidate = body["best_known"]
    tape = "anaideripsninememossomemeninspirediana"
    digest = hashlib.sha256(tape.encode("ascii")).hexdigest()
    assert candidate["rendered"] == "An aide rips nine memos; some men inspire Diana."
    assert candidate["audit"]["exact"] is True
    assert candidate["audit"]["independent_two_pointer"] is True
    assert candidate["audit"]["sha256_forward"] == digest
    assert candidate["audit"]["sha256_reverse"] == digest
    assert candidate["provenance"]["source"] == "project construction run; not catalogue text"
    assert candidate["promotion_status"] == "gated_pending_blinded_readers"
    assert candidate["rlaif"]["certifies_readability"] is False


def test_v4_evaluate_returns_repair_feedback_without_certifying_readability():
    response = client.post("/api/v4/evaluate", json={"text": "An aide rips nine memos; some men inspire Diana."})
    assert response.status_code == 200
    body = response.json()
    assert body["candidate"]["audit"]["exact"] is True
    assert body["rlaif"]["status"] == "diagnostic_only"
    assert body["rlaif"]["human_evidence_required"] is True
    assert body["promotion"]["reader_status"] == "not_run"
    assert "Shakespearean" in body["rlaif"]["framework"]


def test_v4_generation_is_fail_closed():
    for method in ("get", "post"):
        response = getattr(client, method)("/api/v4/generate")
        assert response.status_code == 503
        assert "blinded human-reader evidence" in response.json()["detail"]
