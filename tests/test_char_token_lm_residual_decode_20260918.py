from experiments.char_token_lm_residual_decode_20260918 import audit, consume, run

def test_residual_consumption_is_incremental():
    assert consume("", "a", "a") == ""
    assert consume("", "ab", "ba") == ""
    assert consume("", "a", "b") is None

def test_run_has_independent_controls_and_no_shortcut():
    result = run(max_states=5000)
    assert result["provenance"]["finished_tape_reversed"] is False
    assert result["provenance"]["catalogue_text_imported"] is False
    controls = [x for x in result["rendered_candidates_and_controls"] if x.get("control")]
    assert len(controls) == 2
    for row in controls:
        assert "sha256_forward" in row["audit"]
        assert "sha256_reverse" in row["audit"]
        assert row["audit"]["two_pointer_exact"] is False

def test_audit_reports_forward_and_reverse():
    x = audit("A man, a plan, a canal: Panama")
    assert x["two_pointer_exact"] is True
    assert x["sha256_forward"] == x["sha256_reverse"]
