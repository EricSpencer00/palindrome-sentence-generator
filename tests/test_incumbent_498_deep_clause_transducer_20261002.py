from experiments.incumbent_498_deep_clause_transducer_20261002 import (
    EXPECTED_SHA256,
    build_payload,
)


def test_deep_clause_transducer_replaces_most_of_parent_outer_tape() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert row["audit"]["letters"] == 534
    assert row["audit"]["sha256_forward"] == EXPECTED_SHA256
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["byte_pointer_exact"]
    assert row["audit"]["project_validator_exact"]
    assert row["removed_inherited_outer_letters"] == 258
    assert row["retained_parent_letters"] == 240


def test_transducer_shifts_boundaries_and_carries_live_residual() -> None:
    payload = build_payload()
    state = payload["search_state"]
    transducer = payload["transducer"]

    assert transducer["bridge_exact"]
    assert transducer["left_bridge"] == "A tub? He"
    assert transducer["right_bridge"] == "Eh, but a"
    assert transducer["shell_reverse_exact"]
    assert state["residual_before_center"] == "won"
    assert state["right_boundary_consumption"] == "Leon|won"
    assert state["residual_after_shell"] == ""
