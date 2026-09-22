import json

from experiments.incumbent_666_stop_spot_comparison_20260922 import (
    CHILD_SHA256,
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


def test_stop_spot_comparison_is_exact_but_pending_review():
    payload = json.loads(OUT.read_text())
    row = next(
        row
        for row in payload["rows"]
        if row["id"] == "stop-spot-comparison-mara-saw-666"
    )
    result = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"]["promoted"] is False
    assert row["promotion_status"]["status"] == "pending_full_text_readability_review"
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == CHILD_SHA256
    assert result["sha_equal"]
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]
    assert len(normalize(NEW_LEFT)) == 63
    assert len(normalize(NEW_RIGHT)) == 63
    assert row["live_seam"]["final_residual"] == ""


def test_stop_spot_comparison_removes_old_cluster_with_complete_clauses():
    payload = json.loads(OUT.read_text())
    row = next(
        row
        for row in payload["rows"]
        if row["id"] == "stop-spot-comparison-mara-saw-666"
    )
    rendered = row["rendered"]

    assert "Mara stops rats. Nora spots a ram. Mara sees rats." not in rendered
    assert "Mara sees Aron; star spots Aron. Star sees Aram." not in rendered
    for clause in (
        "Mara saw Noel live.",
        "Nadia stops Aram.",
        "Nadia sees Aram.",
        "Aidan sees Ira.",
        "Ira saw Dog.",
        "God was Ari.",
        "Ari sees Nadia.",
        "Mara sees Aidan.",
        "Mara spots Aidan.",
        "Evil Leon was Aram.",
    ):
        assert clause in rendered
    assert row["readability_delta"]["complete_finite_clauses"]
    assert not row["readability_delta"]["predicate_less_fragment"]
