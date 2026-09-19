from experiments.palindrome_grammar_reachability_20260918 import Grammar, analyze, make_grammar


def finite_language(words):
    g = Grammar()
    g.end = g.node()
    for word in words:
        g.phrase(g.start, g.end, word)
    return g


def test_finite_language_matches_independent_enumeration():
    words = ['ab', 'aba', 'abba', 'abcba', 'abc', 'baab', 'baba']
    report = analyze(finite_language(words))
    actual = {w['text'].strip('.') for w in report['witnesses']}
    assert actual == {w for w in words if w == w[::-1]}
    assert not report['unbounded_exact_language']


def test_word_path_cannot_switch_in_middle():
    # Switching from abc to xba after the first character would invent aba.
    report = analyze(finite_language(['abc', 'xba']))
    assert not report['nonempty_exact_language']


def test_seed_withholding_removes_all_nonempty_solutions_at_every_length():
    seed = analyze(make_grammar())
    assert seed['unbounded_exact_language']
    assert any(w['audit']['letters'] == 38 for w in seed['witnesses'])
    withheld = analyze(make_grammar(True, True))
    assert not withheld['nonempty_exact_language']
    assert not withheld['unbounded_exact_language']
