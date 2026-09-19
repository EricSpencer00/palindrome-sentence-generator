"""Independent brute-force existence oracles for packed character paths."""
import itertools
import random

from experiments.palindromic_language_reachability_20260919 import (
    compile_templates, letters, solve,
)


def test_packed_solver_matches_exhaustive_languages():
    rng = random.Random(20260919)
    for _ in range(100):
        templates = [tuple(tuple("".join(rng.choice("abc") for _ in range(rng.randint(1, 4)))
                                 for _ in range(rng.randint(1, 3)))
                           for _ in range(rng.randint(1, 4)))
                     for _ in range(rng.randint(1, 3))]
        limit = rng.randint(1, 17)
        expected = set()
        language = set()
        for slots in templates:
            for words in itertools.product(*slots):
                tape = "".join(words)
                language.add(tape)
                if len(tape) <= limit and tape == tape[::-1]:
                    expected.add(len(tape))
        result = solve(compile_templates(templates), max_letters=limit)
        rows = result["representative_exact_candidates"]
        assert {r["audit"]["letters"] for r in rows} == expected
        assert all(letters(r["rendered"]) in language for r in rows)


def test_center_can_cross_word_boundary_or_be_inside_word():
    templates = [(("ab",), ("a",)), (("a",), ("ba",)), (("abba",),)]
    result = solve(compile_templates(templates))
    assert {r["audit"]["letters"] for r in result["representative_exact_candidates"]} == {3, 4}


def test_disjoint_outer_letters_prove_empty_without_enumeration():
    slots = (("the baker", "a sailor"), ("sees", "carries"), ("a map", "the letter"))
    result = solve(compile_templates([slots]))
    assert result["paired_states"] == 1
    assert result["representative_exact_candidates"] == []
    assert result["dead_end_certificates"][0]["forward_next"] == ["a", "t"]
    assert result["dead_end_certificates"][0]["backward_next"] == ["p", "r"]
