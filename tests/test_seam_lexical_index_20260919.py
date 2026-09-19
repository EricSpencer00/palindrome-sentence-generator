from experiments.seam_lexical_index_20260919 import independent_audit, search

def test_independent_audit_detects_exact_tape():
    row = independent_audit("A man, a plan, a canal: Panama")
    assert row["exact"] and row["sha256_equal"] and row["two_pointer"]

def test_seam_index_is_reproducible_and_does_not_claim_unfound_hits():
    result = search()
    assert result["method"] == "authored_clause_residual_index"
    assert result["joined_pairs_checked"] == 0
    assert result["exact_hits"] == []
    assert result["independent_audit"].startswith("ASCII letters")
