import json

from experiments.incumbent_666_semantic_role_graft_20260922 import (
    CHILD_SHA256,
    NEW_LEFT,
    NEW_RIGHT,
    OUT,
    PARENT_SHA256,
    independent_audit,
    normalize,
)


def test_semantic_role_graft_is_independently_exact_with_empty_residual():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "semantic-role-graft-mara-nadia-666")
    result = independent_audit(row["rendered"])

    assert row["parent_sha256"] == PARENT_SHA256
    assert row["promotion_status"]["promoted"] is False
    assert result["normalized_letters"] == 666
    assert result["two_pointer_exact"]
    assert result["sha256_forward"] == CHILD_SHA256
    assert result["sha_equal"]
    assert normalize(NEW_LEFT) == normalize(NEW_RIGHT)[::-1]
    assert row["live_seam"]["final_residual"] == ""
    assert len(row["live_seam"]["left_trace"]) == 70
    assert len(row["live_seam"]["right_trace"]) == 70


def test_semantic_role_graft_has_varied_complete_svo_relations():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "semantic-role-graft-mara-nadia-666")
    roles = row["semantic_roles"]

    assert roles["varied_relations"] == ["sees", "stops", "spots"]
    assert roles["repeated_neighboring_clauses"] is True
    assert roles["duplicate_boundary_evidence"] == {
        "left": {
            "normalized_cursor": 127,
            "parent_clause": "Nora sees Aram.",
            "graft_clause": "Nora sees Aram.",
        },
        "right": {
            "normalized_cursor": 539,
            "graft_clause": "Mara sees Aron.",
            "parent_clause": "Mara sees Aron;",
        },
    }
    assert roles["complete_svo_clauses"] is True
    assert roles["vocative_or_appositive_fragments"] is False
    assert row["readability_delta"]["repeated_saw_noel_live_before"] == 2
    assert row["readability_delta"]["repeated_saw_noel_live_after"] == 0
    assert row["readability_delta"]["material_full_text_improvement"] is False
