import json

from experiments.incumbent_602_fragment_seam_repair_20260922 import (
    FRONTIER,
    OUT,
    PARENT_SHA256,
    build_payload,
    independent_audit,
)


def test_fragment_repair_is_exact_and_longer_than_568():
    payload = build_payload()
    row = payload["rows"][0]
    assert payload["stats"]["children_at_least_602"] == 1
    assert payload["preserved_frontier"] == list(FRONTIER)
    assert all({"artifact", "id", "letters", "sha256"} <= set(entry) for entry in FRONTIER)
    assert row["audit"]["letters"] == 620
    assert row["independent_audit"]["two_pointer_exact"]
    assert row["independent_audit"]["sha_equal"]
    assert row["parent_sha256"] == PARENT_SHA256
    assert row["live_seam"]["final_residual"] == ""
    assert row["live_seam"]["committed_character_contradictions"] == 0
    assert "Nora spots a ram. rats." not in row["rendered"]
    assert "a star Mara stops Aron. Star spots Aram." not in row["rendered"]
    assert row["syntax_repair"]["remaining_debt"]


def test_artifact_reproduces_and_independent_audit_agrees():
    payload = json.loads(OUT.read_text())
    assert payload == build_payload()
    assert [entry["id"] for entry in payload["preserved_frontier"]] == [
        "outer-causal-scene-568-working-incumbent",
        "central-distinct-events-560",
        "typed-center-25",
        "depth39-longest-f1g1h1r",
    ]
    assert independent_audit(payload["rows"][0]["rendered"])["two_pointer_exact"]
