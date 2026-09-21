from experiments.minimum_grammar_rule_repair_20260921 import run


def test_activation_masks_cross_check_exact_solver():
    result = run()
    assert result["stats"]["activation_masks"] == 64
    assert result["stats"]["exact_masks"] > 0
    assert result["stats"]["minimum_activation_cost"] == 2
    assert result["stats"]["fresh_reader_exact_over38"] == 0


def test_seed_is_calibration_only_and_base_is_certified_unsat():
    result = run()
    assert result["base_certificate"]["exact_paths"] == 0
    assert result["calibration_exact_paths"]
    assert all(row["calibration_only"] for row in result["calibration_exact_paths"])
    assert result["synthetic_positive_fixture"]["solver_exact"]
    assert result["synthetic_positive_fixture"]["reader_candidate"] is False
