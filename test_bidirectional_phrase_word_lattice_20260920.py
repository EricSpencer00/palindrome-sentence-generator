from bidirectional_phrase_word_lattice_20260920 import audit, run


def test_independent_audit_detects_exact_and_mismatch():
    assert audit("A man, a plan, a canal: Panama!")["pointer_exact"]
    assert audit("the quiet harbor")["pointer_exact"] is False


def test_run_has_intact_controls_and_provenance():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["rendered_controls"] == 16
    assert result["stats"]["online_states"] > 0
    assert result["controls"]
    assert all(row["provenance"]["finished_tape_reversal"] is False for row in result["controls"])

