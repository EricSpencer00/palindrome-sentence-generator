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

def test_bounded_csp_matches_bruteforce_at_each_length():
    csp = constrained_paths(lengths=range(1, 21))
    expected = {t for t in brute_force(GRAMMAR, ATOMIC) if independent_audit(t)["letters"] <= 20}
    got = {r["rendered"].rstrip(".") for r in csp["paths"]}
    assert got == {t for t in expected if independent_audit(t)["exact"]}
    assert csp["stats"]["pruned"] >= 0

def test_center_inside_word_and_asymmetric_boundaries_are_supported():
    # "level" has an interior center for N=5; no word-boundary symmetry is assumed.
    lex = (Word("level", "N"),)
    grammar = {"S": (("N",),)}
    result = constrained_paths(lexicon=lex, lengths=[5], max_words=1, grammar=grammar)
    assert any(r["rendered"].startswith("level") for r in result["paths"])
