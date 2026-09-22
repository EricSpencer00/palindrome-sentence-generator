import json

from experiments.incumbent_620_repeated_event_repair_20260922 import (
    FRONTIER,
    OUT,
    PARENT_SHA256,
    build_payload,
    independent_audit,
)


def test_repeated_event_repair_is_exact_and_improves_counts():
    payload = build_payload()
    row = payload["rows"][0]
    assert payload["stats"]["children_at_least_620"] == 1
    assert payload["preserved_frontier"] == list(FRONTIER)
    assert row["audit"]["letters"] == 644
    assert row["independent_audit"]["two_pointer_exact"]
    assert row["independent_audit"]["sha_equal"]
    assert row["parent_sha256"] == PARENT_SHA256
    assert row["live_seam"]["final_residual"] == ""
    assert row["live_seam"]["committed_character_contradictions"] == 0
    assert row["repetition_delta"]["Mara stops rats"] == {"before": 3, "after": 2}
    assert row["repetition_delta"]["star spots Aram"] == {"before": 3, "after": 2}


def test_artifact_reproduces_and_audits_independently():
    payload = json.loads(OUT.read_text())
    assert payload == build_payload()
    assert independent_audit(payload["rows"][0]["rendered"])["two_pointer_exact"]
