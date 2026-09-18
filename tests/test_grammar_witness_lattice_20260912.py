"""Exact intersection and complete sentence witnesses, not readability tests."""
from __future__ import annotations

import itertools
import random

from experiments.grammar_witness_lattice_20260912 import (
    FRAMES, Lattice, alt, independent_exact_audit, letters, lit, menu, run,
    sentence_witness, seq, solve,
)


def enumerate_paths(lattice, node=None, path=()):
    node = lattice.start if node is None else node
    if node == lattice.end:
        yield path
    for edge in lattice.outgoing[node]:
        yield from enumerate_paths(lattice, edge.target, path + (edge.identifier,))


def brute(lattice, lower=0, upper=100):
    results = set()
    for path in enumerate_paths(lattice):
        text = lattice.render(path)
        tape = letters(text)
        units = text.lower().split()
        if (tape and tape == tape[::-1] and lower <= len(tape) <= upper
                and len(units) == len(set(units))
                and all(letters(word) != letters(word)[::-1] for word in units)):
            results.add(text)
    return sorted(results)


def test_graph_intersection_matches_bruteforce_with_uneven_multiword_branches():
    # Synthetic lexical strings are algorithm controls only, never candidates.
    lattice = Lattice(alt("routes",
        seq("crossing", lit("left", "ab"), lit("right", "cdcba")),
        seq("split", lit("left", "ab cd"), lit("right", "cba")),
        seq("unequal", menu("opening", ("ab", "abc", "bc")),
            menu("middle", ("cd", "cb", "ca")), menu("ending", ("ba", "dba", "cb")))))
    result = solve(lattice, min_letters=0, max_letters=100, record_rejections=False)
    assert result["solutions"] == brute(lattice)
    assert "Ab cdcba" in result["solutions"]
    assert result["stats"]["cross_phrase_cancellations"] > 0
    assert all(row["independent_exact_audit"]["exact"] for row in result["exact_closures"])
    assert all(not row["intact_grammatical_sentence_witness"] for row in result["exact_closures"])


def test_incompatible_voice_branches_cannot_be_spliced_into_false_closure():
    lattice = Lattice(alt("voice",
        seq("first", lit("subject", "ab"), lit("verb", "cd")),
        seq("second", lit("subject", "ef"), lit("verb", "ba"))))
    result = solve(lattice, min_letters=0, max_letters=100, record_rejections=False)
    assert result["solutions"] == []
    assert lattice.parse("Ab ba") is None
    assert result["stats"]["incompatible_grammar_branches"] > 0


def test_random_grammars_match_exhaustive_path_enumeration():
    rng = random.Random(20260912)
    vocab = ["".join(chars) for size in (2, 3) for chars in itertools.product("abc", repeat=size)]
    for _ in range(25):
        lattice = Lattice(alt("topology", *(
            seq(f"branch{branch}", *(menu(f"role{slot}", rng.sample(vocab, 3))
                                    for slot in range(rng.randint(2, 4))))
            for branch in range(3))))
        assert solve(lattice, min_letters=0, max_letters=100,
                     record_rejections=False)["solutions"] == brute(lattice)


def test_empty_rewrite_and_length_bounds_do_not_admit_short_closure():
    lattice = Lattice(seq("sentence", lit("opening", "ab"),
                         menu("optional", ("", "cd")), lit("ending", "ba")))
    assert solve(lattice, min_letters=5, max_letters=100,
                 record_rejections=False)["solutions"] == []
    assert solve(lattice, min_letters=0, max_letters=4,
                 record_rejections=False)["solutions"] == ["Ab ba"]


def test_full_surface_witness_rejects_fragments_and_role_splices():
    frame = FRAMES[0]
    lattice = Lattice(frame.grammar())
    assert sentence_witness(lattice, frame.source, frame)["intact"]
    assert not sentence_witness(lattice, frame.source, FRAMES[1])["intact"]
    assert not sentence_witness(lattice, "Regional volunteers warm blankets.", frame)["intact"]
    assert not sentence_witness(lattice, frame.source[:-1], frame)["intact"]
    assert not sentence_witness(lattice, frame.source + " Quiet archives.", frame)["intact"]
    assert lattice.parse("Warm blankets are delivered to tired evacuees by we because displaced families need protection from bitter weather.") is None
    assert lattice.parse("Warm blankets are delivered to tired evacuees by us because displaced families need protection from bitter weather.") is not None


def test_bounded_run_preserves_complete_rendered_rejections_and_central_checks():
    result = run(max_states=1000)
    assert result["config"]["complete_grammar_path_required_during_search"]
    assert not result["config"]["catalogue_used_for_generation"]
    assert result["readable_survivors"] == []
    for search in result["runs"]:
        assert search["source_control"]["intact_grammatical_sentence_witness"]
        assert search["source_control"]["independent_exact_audit"]["letters"] >= 100
        assert search["hard_rejections"]
        if search["exhausted_frozen_grammar"] and not search["exact_closures"]:
            assert sum(row["unexpanded_middle_derivations"] for row in search["hard_rejections"]) == search["grammar_derivations"]
        for row in search["hard_rejections"]:
            assert row["rendered"].endswith(".")
            assert row["intact_grammatical_sentence_witness"]
            assert row["sentence_witness"]["independent_surface_parse"]
            assert row["current_central_admission"]
            assert row["hard_rejection"] in row["rejection_codes"]
            assert not row["mechanically_admitted"]
            assert independent_exact_audit(row["rendered"]) == row["independent_exact_audit"]


def test_state_cap_is_reported_as_incomplete_not_exhaustive():
    lattice = Lattice(seq("sentence", menu("opening", ("ab", "bc")), menu("ending", ("ba", "cb"))))
    result = solve(lattice, min_letters=0, max_letters=100, max_states=1, record_rejections=False)
    assert not result["exhausted_frozen_grammar"]
    assert result["stats"]["states"] == 1


def test_independent_audit_reports_shifted_boundaries_and_unicode_rejection():
    report = independent_exact_audit("Ab cdcba")
    assert report["exact"] and report["letters"] == 7
    assert report["shifted_word_boundaries"] == [2]
    assert not independent_exact_audit("Ab cdcbá")["exact"]
