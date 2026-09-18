from experiments.asymmetric_bridge_attachment_20260918 import audit, run, seam_residual

def test_independent_audit_is_not_constructor_score():
    row = audit("An aide rips nine memos; some men inspire Diana.")
    assert row["exact"] and row["independent_two_pointer"] and row["hashes_equal"]

def test_bridge_equation_is_checked_before_rendering():
    x = seam_residual("the baker marks fresh maps", "Mara", "the writer opens the gate", "at dawn", "right")
    assert "required_reverse_prefix" in x and "bridge_tape" in x
    assert x["equation_satisfied"] is False

def test_run_has_complete_provenance_and_no_shortcut_rows():
    result = run()
    assert result["stats"]["rendered"] == 12
    assert result["stats"]["exact"] == 0
    for row in result["rendered_candidates"]:
        assert row["novelty_preflight"]["fragment"] is False
        assert row["provenance"]["catalogue_used"] is False
