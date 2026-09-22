from experiments.incumbent_534_inner_event_repair_20261002 import build_payload


def test_inner_repairs_produce_exact_frontier_over_530() -> None:
    payload = build_payload()

    assert payload["stats"] == {
        "authored_repair_paths": 4,
        "independently_exact_children": 4,
        "children_over_530": 4,
        "shortest_letters": 536,
        "longest_letters": 544,
    }
    for row in payload["rows"]:
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["byte_pointer_exact"]
        assert row["audit"]["project_validator_exact"]
        assert row["audit"]["sha_equal"]
        assert row["live_state"]["residual_before_center"] == "won"
        assert row["live_state"]["residual_after_shell"] == ""
        assert all(repair["replacement_reverse_exact"] for repair in row["repairs"])


def test_combined_child_repairs_two_distinct_inner_seams() -> None:
    payload = build_payload()
    row = next(item for item in payload["rows"] if item["id"] == "combined-finite-role-544")

    assert row["audit"]["letters"] == 544
    assert row["audit"]["sha256_forward"] == (
        "2ea2411e5fea4d27d3db24ba0e471cc6a52a4a2196b52fb6a82b5cc6033b0655"
    )
    assert [repair["id"] for repair in row["repairs"]] == [
        "finite-delivery",
        "role-event",
    ]
    assert "Nadia delivers maps. Leon," in row["rendered"]
    assert "“Evil Leon” was Aron." in row["rendered"]
