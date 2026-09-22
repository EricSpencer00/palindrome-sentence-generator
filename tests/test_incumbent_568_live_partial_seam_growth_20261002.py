from experiments.incumbent_568_live_partial_seam_growth_20261002 import build_payload


def test_live_partial_seam_produces_diverse_exact_growth() -> None:
    payload = build_payload()
    rows = payload["rows"]

    assert len(rows) == 4
    assert len({row["audit"]["sha256_forward"] for row in rows}) == 4
    assert all(row["audit"]["letters"] > 568 for row in rows)
    assert all(row["audit"]["two_pointer_exact"] for row in rows)
    assert all(row["audit"]["byte_pointer_exact"] for row in rows)
    assert all(row["audit"]["project_validator_exact"] for row in rows)
    assert all(row["live_seam"]["final_residual"] == "" for row in rows)
    assert all(row["live_seam"]["backtracks"] == 0 for row in rows)
    assert len({tuple(row["new_event_content"]) for row in rows[:3]}) == 3


def test_repaired_child_is_new_incumbent_and_reduces_repetition() -> None:
    payload = build_payload()
    repaired = payload["rows"][-1]

    assert repaired["id"] == "live-nadia-seam-shell-repair-608"
    assert repaired["working_status"] == "working_length_incumbent"
    assert repaired["audit"]["letters"] == 608
    assert payload["working_length_incumbent"]["sha256"] == repaired["audit"]["sha256_forward"]
    assert all(
        delta == {"before": 2, "after": 1}
        for delta in repaired["shell_repair"]["repetition_delta"].values()
    )
    assert [item["letters"] for item in payload["preserved_frontier"]] == [568, 560, 556]
    assert "do not reject" in repaired["repair_debt"]["effect"]
