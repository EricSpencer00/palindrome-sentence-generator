import itertools

from experiments.discourse_subset_word_equation_20260914 import (
    cancel, independent_audit, letters, solve,
)


def test_residual_crosses_multiple_sentence_boundaries():
    clauses = ("abc d", "c", "ba")
    found, stats = solve(clauses, min_letters=7, max_letters=7, max_clauses=3, state_budget=1000)
    assert (0, 1, 2) in found
    assert independent_audit("abc d c ba")["exact"]
    assert stats["max_sentences_in_state"] == 3


def test_matches_brute_force_for_boundary_disjoint_tiny_fixture():
    clauses = ("abcd", "c", "ba", "q")
    expected = set()
    for count in range(1, 5):
        for order in itertools.permutations(range(len(clauses)), count):
            tape = "".join(clauses[i] for i in order)
            if len(tape) == 7 and tape == tape[::-1]:
                expected.add(order)
    found, stats = solve(clauses, min_letters=7, max_letters=7, max_clauses=4, state_budget=10000)
    assert set(found) == expected
    assert not stats["budget_exhausted"]


def test_cancel_tracks_correct_residual_owner():
    assert cancel("abcdef", "abc") == ("def", 1)
    assert cancel("ab", "abcde") == ("cde", -1)
    assert cancel("ab", "ax") is None


def test_independent_audit_recomputes_mismatches_and_unicode_rejection():
    assert independent_audit("Abc, d cba!")["exact"]
    assert independent_audit("Abc, d eba!")["mismatched_pair_count"] == 1
    assert not independent_audit("éabaé")["exact"]
    assert letters("A-b, C!") == "abc"
