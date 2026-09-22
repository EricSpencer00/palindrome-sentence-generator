import json

from experiments.incumbent_644_scaffold_repair_20260922 import (
    NEXT_OPERATOR,
    OUT,
    SUPERSEDING_REPAIR_ARTIFACT,
    SUPERSEDING_REPAIR_ID,
    independent_audit,
)


def test_652_row_is_rejected_for_fragment_debt_and_points_to_completed_repair():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "scaffold-repair-aidan-draws-652")

    assert row["working_status"] == "rejected_non_promoted_for_fragment_debt"
    assert row["promotion_status"]["status"] == "rejected_non_promoted"
    assert row["promotion_status"]["reason_code"] == "fragment_debt"
    assert "dangling vocative/appositive" in row["promotion_status"]["reason"]
    assert row["promotion_status"]["superseded_by"] == {
        "artifact": SUPERSEDING_REPAIR_ARTIFACT,
        "id": SUPERSEDING_REPAIR_ID,
        "status": "completed_same_window_clause_repair",
    }
    assert row["next_operator"] == NEXT_OPERATOR
    assert payload["next_operator"] == NEXT_OPERATOR


def test_652_exact_artifact_evidence_is_unchanged_and_independently_verified():
    payload = json.loads(OUT.read_text())
    row = next(row for row in payload["rows"] if row["id"] == "scaffold-repair-aidan-draws-652")
    audit = independent_audit(row["rendered"])

    assert audit["normalized_letters"] == 652
    assert audit["two_pointer_exact"]
    assert audit["sha256_forward"] == (
        "0b7f1950beac54b34a8ec66e5150f995abeda4d2d07bfd2f0eb454fed108f1e5"
    )
    assert audit["sha_equal"]
    assert row["audit"]["sha256_forward"] == audit["sha256_forward"]

