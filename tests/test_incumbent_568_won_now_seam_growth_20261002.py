from experiments.incumbent_568_won_now_seam_growth_20261002 import build_payload


def test_won_now_seam_produces_new_exact_568_lineage_children() -> None:
    payload = build_payload()
    rows = payload["rows"]

    assert payload["working_incumbent"]["letters"] == 568
    assert payload["working_incumbent"]["sha256"] == (
        "6647fe46becb64b0841785f0bd9865070254888b449be228d22cfbedeb1e0380"
    )
    assert len(rows) == 3
    assert len({row["audit"]["sha256_forward"] for row in rows}) == 3
    assert all(row["audit"]["letters"] > 568 for row in rows)
    assert all(row["audit"]["two_pointer_exact"] for row in rows)
    assert all(row["audit"]["byte_pointer_exact"] for row in rows)
    assert all(row["audit"]["project_validator_exact"] for row in rows)
    assert all(row["live_seam"]["left_partial_join"] == "w|on" for row in rows)
    assert all(row["live_seam"]["right_partial_join"] == "no|w" for row in rows)
    assert all(row["live_seam"]["final_residual"] == "" for row in rows)
    assert all(row["live_seam"]["committed_character_contradictions"] == 0 for row in rows)


def test_won_now_seam_preserves_required_frontier_and_records_debt() -> None:
    payload = build_payload()

    assert [item["letters"] for item in payload["preserved_frontier"]] == [568, 560, 558, 556]
    assert all(len(row["new_event_content"]) == 2 for row in payload["rows"])
    assert all(row["repair_debt"]["inherited_proper_spans"] for row in payload["rows"])
    assert all("do not demote" in row["repair_debt"]["effect"] for row in payload["rows"])
