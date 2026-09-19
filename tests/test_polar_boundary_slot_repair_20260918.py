from experiments.polar_boundary_slot_repair_20260918 import audit, forbidden_hidden, run


def test_hidden_span_filter_rejects_boundary_shortcut():
    assert forbidden_hidden("Was Noel a gas?, saga, Leon saw.")


def test_bounded_polar_lane_has_provenance_and_exact_audits():
    result = run(max_probes=100)
    assert result["stats"]["stored_probes"] == 100
    assert result["provenance"]["direct_reversible_word_pairs_forbidden"]
    assert result["provenance"]["hidden_palindromic_spans_forbidden"]
    for row in result["exact_candidates"]:
        assert row["audit"]["two_pointer_exact"]
        assert row["audit"]["sha_equal_under_reversal"]
        assert row["forbidden_hidden_spans"]
