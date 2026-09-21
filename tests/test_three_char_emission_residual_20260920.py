from experiments.three_char_emission_residual_20260920 import run


def test_key_is_derived_from_emitted_outer_chunks():
    result = run()
    assert result["derived_key_trace"] == ["aca", "our", "the"]
    assert all(row["provenance"]["key_derived_from_outer_emission"] for row in result["diagnostic_candidates"])


def test_three_char_run_is_diagnostic_and_records_repair():
    result = run()
    assert result["stats"]["exact_above_38"] == 0
    assert result["reader_facing_candidates"] == []
    assert result["next_repair"]["operator"] == "typed continuation by residual suffix class"
