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
    assert body["optimization"]["objective_order"][2] == "longer rendered tape"


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
    assert candidate["provenance"]["run_id"] == "half-tape-grammar-csp-20260919"
    assert candidate["provenance"]["search_summary"]["pilot_lengths"] == "40–52; no exact closure"
    assert "longest 50 letters" in candidate["provenance"]["search_summary"]["latest_dream_rsi_repair"]
    assert candidate["promotion_status"] == "gated_pending_blinded_readers"
    assert candidate["rlaif"]["certifies_readability"] is False
    frontier = body["repair_frontier"]
    assert frontier[0]["letters"] == 50
    assert frontier[0]["exact"] is True
    assert frontier[0]["mechanically_admitted"] is True
    assert frontier[0]["reader_status"] == "not_run"
    assert frontier[1]["letters"] == 66
    assert frontier[1]["mechanically_admitted"] is False
    assert "hidden proper palindrome span" in frontier[1]["rejection"]


def test_v4_evaluate_returns_repair_feedback_without_certifying_readability():
    response = client.post("/api/v4/evaluate", json={"text": "An aide rips nine memos; some men inspire Diana."})
    assert response.status_code == 200
    body = response.json()
    assert body["candidate"]["audit"]["exact"] is True
    assert body["rlaif"]["status"] == "diagnostic_only"
    assert body["rlaif"]["human_evidence_required"] is True
    assert body["promotion"]["reader_status"] == "not_run"
    assert "Shakespearean" in body["rlaif"]["framework"]
    assert body["rlaif"]["scores"]["dramatic_cadence_diagnostic"] > 0
    assert body["rlaif"]["repairs"]


def test_v4_method_and_best_evaluation_are_explicitly_diagnostic():
    method = client.get("/api/v4/method")
    assert method.status_code == 200
    assert method.json()["status"] == "constructive_search_in_progress"
    assert method.json()["current_best"]["rendered"] == "An aide rips nine memos; some men inspire Diana."
    assert method.json()["optimization"]["current_search"] == "dream-rsi-strict-phrase-bank-20260919"
    assert method.json()["optimization"]["search_history"][-1] == "dream-rsi-strict-phrase-bank-20260919"
    assert method.json()["repair_frontier"][0]["letters"] == 50

    evaluation = client.get("/api/v4/best-evaluation")
    assert evaluation.status_code == 200
    assert evaluation.json()["candidate"]["audit"]["exact"] is True
    assert evaluation.json()["rlaif"]["certifies_readability"] is False


def test_v4_generation_is_fail_closed():
    for method in ("get", "post"):
        response = getattr(client, method)("/api/v4/generate")
        assert response.status_code == 503
        assert "blinded human-reader evidence" in response.json()["detail"]
