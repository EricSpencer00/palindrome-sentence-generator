import json

from experiments.incumbent_666_comparison_alternative_20260922 import (
    CHILD_SHA256,
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


def test_comparison_alternative_is_exact_but_not_promoted():
    payload = json.loads(OUT.read_text())
    row = next(
        row
        for row in payload["rows"]
        if row["id"] == "comparison-alternative-nora-sees-666"
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
    assert len(normalize(NEW_LEFT)) == 44
    assert len(normalize(NEW_RIGHT)) == 44
    assert row["live_seam"]["final_residual"] == ""


def test_comparison_alternative_removes_targeted_cluster_with_finite_clauses():
    payload = json.loads(OUT.read_text())
    row = next(
        row
        for row in payload["rows"]
        if row["id"] == "comparison-alternative-nora-sees-666"
    )
    rendered = row["rendered"]

    assert "Spam's reviled, Nadia" not in rendered
    for clause in (
        "Nora sees Nadia.",
        "Nadia sees Ira.",
        "Sara saw God.",
        "Ari saw Dog.",
        "God was Ira.",
        "Dog was Aras.",
        "Ari sees Aidan.",
        "Aidan sees Aron.",
    ):
        assert clause in rendered
    assert row["readability_delta"]["complete_finite_clauses"]
    assert not row["readability_delta"]["predicate_less_fragment"]

