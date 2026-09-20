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
    small = ATOMIC[:9]
    csp = constrained_paths(lexicon=small, lengths=range(1, 21), max_nodes=20000)
    expected = {t for t in brute_force(GRAMMAR, small, max_words=8) if independent_audit(t)["letters"] <= 20 and admission_ok(t.split())}
    got = {letters(r["rendered"]) for r in csp["paths"]}
    assert got == {letters(t) for t in expected if independent_audit(t)["exact"]}
    assert csp["stats"]["pruned"] >= 0

def test_atomic_38_letter_witness_is_recovered():
    anchor_lex = tuple(w for w in ATOMIC if w.text in {"an", "aide", "rips", "nine", "memos", "some", "men", "inspire", "diana"})
    result = constrained_paths(lexicon=anchor_lex, lengths=[38], max_words=10, max_nodes=20000)
    anchor = "an aide rips nine memos some men inspire diana"
    assert any(letters(r["rendered"]) == letters(anchor) for r in result["paths"])
    assert any(";" in r["rendered"] for r in result["paths"] if letters(r["rendered"]) == letters(anchor))
    assert independent_audit(anchor)["exact"]

def test_shortcut_witnesses_are_rejected():
    rows = constrained_paths(lengths=[38], max_words=10, max_nodes=1000)["paths"]
    assert all("ava sees ava ava sees ava" not in r["rendered"] for r in rows)
    assert all(len(set(r["rendered"].replace(";", "").rstrip(".").split())) == len(r["rendered"].replace(";", "").rstrip(".").split()) for r in rows)

def test_center_inside_word_and_asymmetric_boundaries_are_supported():
    # The center of ``a bcba`` falls inside the second word; no boundary
    # symmetry is assumed and neither lexical unit is a shortcut palindrome.
    lex = (Word("a", "N"), Word("bcba", "N"))
    grammar = {"S": (("N", "N"),)}
    result = constrained_paths(lexicon=lex, lengths=[5], max_words=2, grammar=grammar)
    assert any(letters(r["rendered"]) == "abcba" for r in result["paths"])

def test_brown_loader_is_bounded_and_deterministic():
    a = load_brown_lexicon(limit=2000)
    assert len(a) == 2000 and a == load_brown_lexicon(limit=2000)

def test_bilateral_edges_propagate_and_audit():
    result = bilateral_lexical_csp((Word("anna", "PROPN"),), max_words=2, max_nodes=50)
    assert result["stats"]["status"] == "UNSAT"

def test_bilateral_rejects_mirrored_token_shortcut():
    result = bilateral_lexical_csp((Word("ab", "N"), Word("ba", "N")), max_words=2, max_nodes=50)
    assert result["candidates"] == []

def test_bilateral_requires_two_parsed_clauses():
    lex = tuple(w for w in ATOMIC if w.text in {"an", "aide", "rips", "nine", "memos", "some", "men", "inspire", "diana"})
    result = bilateral_lexical_csp(lex, max_words=10, max_nodes=100000)
    assert result["stats"]["status"] in {"UNSAT", "timeout"}
