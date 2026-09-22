from experiments.incumbent_596_repeated_shell_repair_20261002 import build_payload


def test_repeated_shell_repair_is_exact_and_stays_on_568_lineage() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert payload["working_incumbent"]["letters"] == 568
    assert row["audit"]["letters"] == 594
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["byte_pointer_exact"]
    assert row["audit"]["project_validator_exact"]
    assert row["live_repair"]["final_residual"] == ""
    assert row["live_repair"]["committed_character_contradictions"] == 0
    assert row["lineage_root_sha256"] == payload["working_incumbent"]["sha256"]


def test_repeated_shell_repair_reduces_targeted_scaffolding() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert all(
        delta == {"before": 2, "after": 1}
        for delta in row["repetition_delta"].values()
    )
    assert len(set(row["new_event_content"])) == 4
    assert payload["preserved_frontier_letters"] == [568, 560, 558, 556]
    assert "without demoting 568" in row["repair_debt"]["effect"]
