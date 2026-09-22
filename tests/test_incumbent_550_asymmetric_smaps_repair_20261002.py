from experiments.incumbent_550_asymmetric_smaps_repair_20261002 import build_payload


def test_asymmetric_window_is_nonpalindromic_but_closes_live_residual() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert row["audit"]["letters"] == 531
    assert row["audit"]["two_pointer_exact"]
    assert row["audit"]["byte_pointer_exact"]
    assert row["audit"]["project_validator_exact"]
    assert not row["replacement"]["new_window_is_palindromic"]
    assert row["live_state"]["residual_before_window"] == "smaps"
    assert row["live_state"]["closure_exact"]
    assert row["live_state"]["residual_after_window"] == ""


def test_asymmetric_window_removes_four_spans_and_creates_none() -> None:
    payload = build_payload()
    row = payload["rows"][0]

    assert payload["stats"]["parent_proper_span_count"] == 67
    assert payload["stats"]["child_proper_span_count"] == 63
    assert payload["stats"]["proper_spans_removed"] == 4
    assert payload["stats"]["proper_spans_created_at_replacement_boundaries"] == 0
    assert row["boundary_audit"]["proper_spans_anchored_at_replacement_boundaries"] == []
    assert not row["boundary_audit"]["globally_shortcut_clean"]
    obstruction = row["fixed_window_obstruction"]
    assert obstruction["strict_accepts_at_fixed_window"] == 0
    assert obstruction["next_editable_window"]["normalized_offsets"] == [261, 550]
    assert obstruction["next_editable_window"]["minimum_new_window_letters_for_531_total"] == 270
    eof_obstruction = row["through_eof_obstruction"]
    assert eof_obstruction["strict_accepts_at_through_eof_cut"] == 0
    assert eof_obstruction["repeated_unit"] == ["delivers", "maps"]
    assert eof_obstruction["next_globally_clean_prefix"]["raw_end"] == 48
    assert eof_obstruction["next_globally_clean_prefix"]["normalized_end"] == 36
