from experiments.incumbent_544_cross_boundary_seam_repair_20261002 import (
    EXPECTED_SHA256,
    build_payload,
)


def test_cross_boundary_repair_is_exact_and_longer() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert row["audit"]["letters"] == 550
    assert row["audit"]["sha256_forward"] == EXPECTED_SHA256
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["byte_pointer_exact"]
    assert row["audit"]["project_validator_exact"]
    assert row["growth_over_parent"] == 6


def test_repair_shifts_word_boundaries_and_keeps_live_residuals() -> None:
    row = build_payload()["rows"][0]

    assert not row["structural_audit"]["word_boundaries_reflect_one_to_one"]
    assert row["live_state"]["inner_residual"] == "tes"
    assert row["live_state"]["inner_closure"] == "Set"
    assert row["live_state"]["outer_residual_before_center"] == "won"
    assert row["live_state"]["residual_after_shell"] == ""
    assert "Pat notes." in row["rendered"]
    assert "Seton, tap." in row["rendered"]
