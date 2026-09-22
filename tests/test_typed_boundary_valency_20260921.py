from experiments.typed_boundary_valency_20260921 import audit, online_admit, run

def test_audit_independent_hashes_and_mismatch():
    result = audit("A man, a plan, a canal: Panama")
    assert result["exact"] and result["sha256_forward"] == result["sha256_reverse"]

def test_boundary_admission_is_character_based():
    assert online_admit(["alpha"], ["a" + "ahpla"])
    assert not online_admit(["alpha"], ["omega"])

def test_run_has_provenance_and_real_prose_controls():
    result = run()
    assert result["stats"]["rendered"] == 54
    assert result["reader_facing_candidates"]
    assert all("audit" in row and "provenance" in row for row in result["reader_facing_candidates"])
