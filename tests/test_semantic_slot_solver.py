import itertools
import random

from experiments.semantic_slot_solver import letters, solve, words


def brute(domains):
    result = []
    for parts in itertools.product(*domains):
        text = " ".join(parts)
        tape, units = letters(text), words(text)
        if (tape == tape[::-1] and len(units) == len(set(units))
                and all(w != w[::-1] for w in units)):
            result.append(text)
    return sorted(set(result))


def test_solver_matches_exhaustive_search_including_centre_crossing():
    domains = [["do", "step", "stressed", "red"],
               ["geese", "on", "desserts", "rum"],
               ["see", "no", "murder"], ["god", "pets", "red"]]
    result, _ = solve(domains, min_letters=0, max_letters=100)
    assert result == brute(domains)
    assert "do geese see god" in result


def test_multiword_domains_and_uneven_residuals():
    domains = [["do geese", "step on", "red rum", "stressed"],
               ["see god", "no pets", "murder", "desserts"]]
    result, _ = solve(domains, min_letters=0, max_letters=100)
    assert result == brute(domains)
    assert "red rum murder" in result


def test_repeated_and_self_palindromic_units_are_excluded():
    for domain in [[["a"], ["a"]], [["ab"], ["ba ab"], ["ba"]], [["noon"]]]:
        result, _ = solve(domain, min_letters=0, max_letters=100)
        assert result == []


def test_length_is_a_hard_gate():
    domains = [["stressed"], ["desserts"]]
    assert solve(domains, min_letters=16, max_letters=16)[0] == ["stressed desserts"]
    assert solve(domains, min_letters=17)[0] == []
    assert solve(domains, min_letters=0, max_letters=15)[0] == []


def test_arbitrary_centre_slot_can_close_without_external_word_boundary():
    domains = [["ab"], ["cdcba"]]
    assert solve(domains, min_letters=0, max_letters=100)[0] == ["ab cdcba"]


def test_random_finite_languages_match_bruteforce():
    rng = random.Random(20260912)
    vocabulary = ["".join(chars) for size in (2, 3, 4)
                  for chars in itertools.product("abc", repeat=size)]
    for _ in range(40):
        domains = [rng.sample(vocabulary, 5) for _ in range(rng.randint(2, 4))]
        assert solve(domains, min_letters=0, max_letters=100)[0] == brute(domains)
