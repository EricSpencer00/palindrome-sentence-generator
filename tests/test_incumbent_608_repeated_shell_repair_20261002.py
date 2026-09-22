from experiments.incumbent_608_repeated_shell_repair_20261002 import build_payload


def test_double_shell_repair_is_exact_and_longer() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert row["audit"]["letters"] == 650
    assert row["growth_over_parent"] == 42
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["byte_pointer_exact"]
    assert row["audit"]["project_validator_exact"]
    assert all(repair["final_residual"] == "" for repair in row["live_repairs"])
    assert all(repair["backtracks"] == 0 for repair in row["live_repairs"])
    assert payload["working_length_incumbent"]["sha256"] == row["audit"]["sha256_forward"]


def test_double_shell_repair_halves_repeated_formulas_and_preserves_frontier() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert all(
        delta == {"before": 4, "after": 2}
        for delta in row["repetition_delta"].values()
    )
    assert len(row["new_event_content"]) == 8
    assert len(set(row["new_event_content"])) == 8
    assert [item["letters"] for item in payload["preserved_frontier"]] == [608, 568, 560, 556]
    assert "do not reject" in row["repair_debt"]["effect"]
