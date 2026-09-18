from experiments.agreement_inflection_center_20260918 import audit, run

def test_independent_audit_and_feature_carrying_states():
    result = run()
    assert result["stats"]["states"] > result["stats"]["rendered"]
    assert result["stats"]["exact"] == 0
    for row in result["rendered_candidates"]:
        assert row["equation_state"]["features"]["left_number"] in {"sg", "pl"}
        assert row["audit"] == audit(row["rendered"])
        assert row["provenance"]["reader_eligible"] is False

def test_no_shortcut_provenance():
    result = run()
    assert result["novelty_preflight"]["duplicate_sweep"] is False
    assert all(not r["provenance"]["finished_tape_reversal"] for r in result["rendered_candidates"])
