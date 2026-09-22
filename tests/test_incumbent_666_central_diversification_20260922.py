import json

from experiments.incumbent_666_central_diversification_20260922 import (
    CHILD_SHA256,
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


def test_reviewer_diversification_preserves_exact_666_lineage():
    payload = json.loads(OUT.read_text())
    row = next(
        row
        for row in payload["rows"]
        if row["id"] == "central-diversification-nadia-spots-666"
    )
    audit = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert audit["normalized_letters"] == 666
    assert audit["two_pointer_exact"]
    assert audit["sha256_forward"] == CHILD_SHA256
    assert audit["sha_equal"]
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]
    assert row["live_seam"]["final_residual"] == ""
    assert row["live_seam"]["committed_character_contradictions"] == 0


def test_reviewer_diversification_reduces_center_sees_count():
    payload = json.loads(OUT.read_text())
    row = next(
        row
        for row in payload["rows"]
        if row["id"] == "central-diversification-nadia-spots-666"
    )
    rendered = row["rendered"]

    for clause in (
        "Nadia sees Mara.",
        "Aidan spots Ira.",
        "Ari sees God.",
        "Dog sees Ira.",
        "Ari stops Nadia.",
        "Aram sees Aidan.",
    ):
        assert clause in rendered
    assert row["readability_delta"] == {
        "central_sees_before": 6,
        "central_sees_after": 4,
        "central_spots_before": 0,
        "central_spots_after": 1,
        "complete_finite_clauses": True,
        "predicate_less_fragment": False,
        "dangling_vocative_or_appositive": False,
    }

