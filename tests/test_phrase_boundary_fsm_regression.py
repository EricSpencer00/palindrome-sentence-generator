from experiments.phrase_boundary_fsm_20260920 import audit, consume_prefix, run

def test_independent_clause_palindromes_do_not_admit_combined_tape():
    left = "A man a plan a canal Panama."
    right = "Madam in Eden I'm Adam."
    assert audit(left)["two_pointer_exact"]
    assert audit(right)["two_pointer_exact"]
    assert not audit(left + " " + right)["two_pointer_exact"]

def test_live_cursor_carries_one_sided_debt_across_constituent_edges():
    assert consume_prefix("abcdef", "abc") == ("def", "")
    assert consume_prefix("abc", "abcdef") == ("", "def")
    assert consume_prefix("abc", "abd") is None

def test_corrected_run_reports_live_traversal_without_promoting_controls():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed_corrected_live_traversal"
    assert "live-debt" in result["signature"]
    assert result["stats"]["visited_transitions"] == 154
    assert result["stats"]["complete_clause_pairs"] == 0
    assert result["candidates"] == []
