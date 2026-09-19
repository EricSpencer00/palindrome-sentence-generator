from experiments.semantic_shell_growth_20260919 import audit, grow

def test_independent_audit_detects_exact_palindrome():
    result = audit("An aide rips nine memos; some men inspire Diana.")
    assert result["two_pointer_exact"] and result["sha_equal"]

def test_shell_growth_is_reproducible_and_not_finished_tape_reversal():
    result = grow(max_depth=1)
    assert result["provenance"]["rlaif_per_candidate"] is False
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["rendered"] > 0
    assert all(row["provenance"]["finished_tape_reversed"] is False for row in result["actual_candidates"])
