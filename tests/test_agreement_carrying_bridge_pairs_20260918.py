from experiments.agreement_carrying_bridge_pairs_20260918 import audit, residual, run

def test_independent_audit_exact_and_nonexact():
    assert audit("A man, a plan, a canal: Panama.")["independent_two_pointer"]
    assert not audit("The baker marks a map.")["independent_two_pointer"]

def test_name_residual_is_solved_before_rendering():
    row = residual("The baker marks a map", "the pilot reads {name}", "Mara")
    assert "required_residual" in row and row["name_tape"] == "mara"
    assert row["compatible"] is False

def test_run_has_provenance_and_no_reader_claim():
    out = run()
    assert out["stats"]["rendered"] == 40
    assert all(r["provenance"]["catalogue_used"] is False for r in out["rendered_candidates"])
    assert all("not certified" in r["reader_status"] for r in out["rendered_candidates"])
