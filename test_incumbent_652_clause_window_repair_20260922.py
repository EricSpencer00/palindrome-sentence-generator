import json

from experiments.incumbent_652_clause_window_repair_20260922 import (
    FRONTIER,
    OUT,
    PARENT_SHA256,
    build_payload,
    independent_audit,
)


def test_same_window_repair_is_exact_and_clause_complete():
    payload = build_payload()
    row = payload["rows"][0]
    assert payload["stats"]["children_at_least_644"] == 1
    assert payload["preserved_frontier"] == list(FRONTIER)
    assert row["audit"]["letters"] == 666
    assert row["independent_audit"]["two_pointer_exact"]
    assert row["independent_audit"]["sha_equal"]
    assert row["parent_sha256"] == PARENT_SHA256
    assert row["live_seam"]["final_residual"] == ""
    assert row["syntax_repair"]["complete_left_clauses"]
    assert row["syntax_repair"]["complete_right_clauses"]
    assert not row["syntax_repair"]["dangling_vocative_or_appositive"]
    assert not row["syntax_repair"]["predicate_less_fragment"]
    assert "Nora sees Aram" in row["new_event_content"]
    assert "Aidan sees Aram" in row["new_event_content"]


def test_artifact_reproduces_and_audits_independently():
    payload = json.loads(OUT.read_text())
    assert payload == build_payload()
    assert independent_audit(payload["rows"][0]["rendered"])["two_pointer_exact"]
