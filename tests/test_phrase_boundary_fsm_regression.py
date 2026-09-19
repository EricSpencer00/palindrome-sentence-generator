from experiments.phrase_boundary_fsm_20260920 import audit

def test_independent_clause_palindromes_do_not_admit_combined_tape():
    left = "A man a plan a canal Panama."
    right = "Madam in Eden I'm Adam."
    assert audit(left)["two_pointer_exact"]
    assert audit(right)["two_pointer_exact"]
    assert not audit(left + " " + right)["two_pointer_exact"]
