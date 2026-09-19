from experiments.dream_rsi_discourse_ellipsis_20260918 import render, row, run


def test_discourse_frame_has_independent_exact_audit():
    candidate = row("Noel", ("smart", "stressed"), ("desserts", "trams"), "Leon")
    assert candidate["audit"]["two_pointer_exact"]
    assert candidate["audit"]["sha_equal_under_reversal"]
    assert candidate["letters"] == 40


def test_strict_gate_rejects_word_mirror_or_proper_palindromic_island():
    candidate = row("Noel", ("smart", "stressed"), ("desserts", "trams"), "Leon")
    assert not all(candidate["mechanical_checks"].values())
    assert candidate["anti_shortcut_flags"]["word_order_only"]
    assert candidate["anti_shortcut_flags"]["self_palindromic_proper_span"]


def test_run_records_exact_rows_but_no_reader_admission():
    result = run()
    assert result["stats"]["exact"] > 0
    assert result["stats"]["mechanically_admitted"] == 0
    assert result["reader_gate"]["status"] == "closed"
