from forward_lexicalized_grammar_20260920 import *

def test_solver_matches_bruteforce_and_audits_independently():
    result = solve()
    expected = brute_force(GRAMMAR, ATOMIC)
    assert result["stats"]["grammar_derivations"] == len(expected)
    assert all(independent_audit(row["rendered"]) == row["audit"] for row in result["near_misses"])

def test_no_injected_anchor_or_half_clause_pairing():
    result = solve()
    assert result["novelty_preflight"]["phrase_injected"] is False
    assert result["novelty_preflight"]["independently_authored_half_clauses"] is False

def test_forward_reverse_hashes_agree_only_for_exact():
    for text in brute_force(GRAMMAR, ATOMIC):
        audit = independent_audit(text)
        assert audit["exact"] == (audit["sha256_forward"] == audit["sha256_reverse"])
