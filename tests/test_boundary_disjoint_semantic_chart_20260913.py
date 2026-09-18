"""Structural toys are non-English kernel fixtures, never candidate prose."""
from itertools import product

from experiments.boundary_disjoint_semantic_chart_20260913 import (
    FRAMES, LAYOUTS, boundary_chart, domains, run, typed_parse,
)
from llm_palindrome.admission import (
    REPEATABLE_FUNCTION_WORDS, has_self_palindromic_proper_multiword_span,
)


def test_shifted_boundaries_keep_a_free_word_internal_center():
    result = boundary_chart((("ab",), ("cd",), ("cba",)))
    assert result["states_exhausted"]
    [record] = result["records"]
    assert record["tape"] == "abcdcba"
    assert record["matched_depth"] == 3
    assert record["center_characters"] == 1
    assert record["left_boundary_depths"] == (2,)
    assert record["right_boundary_depths"] == (3,)
    assert not has_self_palindromic_proper_multiword_span(tuple(record["words"]))


def test_proper_multiword_island_is_pruned_before_its_center():
    slots = (("ab",), ("cd",), ("dc",), ("ba",))
    baseline = boundary_chart(slots, prune_islands=False, prune_content_center=False)
    assert baseline["records"][0]["tape"] == "abcddcba"
    result = boundary_chart(slots)
    assert result["records"] == []
    assert result["stats"]["proper_island_prunes"] == 1
    [witness] = result["prune_witnesses"]
    assert witness["matched_depth"] == 2
    assert witness["interior_token_count"] == 2
    assert witness["left_boundary_depths"] == witness["right_boundary_depths"] == (2,)
    assert result["stats"]["states"] < baseline["stats"]["states"]


def test_pruning_matches_exhaustive_reference_on_small_branching_grammars():
    grammars = (
        (("ab", "a"), ("cd", "bc"), ("cba", "ba")),
        (("ab", "a"), ("cd", "d"), ("dc", "d"), ("ba", "a")),
        (("abc", "a"), ("d", "bc"), ("ba", "cba")),
    )
    for slots in grammars:
        # The chart's online test only rejects globally centered islands;
        # off-center islands remain the unchanged final gate's responsibility.
        expected = set()
        for words in product(*slots):
            tape = "".join(words)
            if tape != tape[::-1]:
                continue
            offsets = [0]
            for word in words:
                offsets.append(offsets[-1] + len(word))
            forbidden = False
            for i in range(1, len(words)):
                for j in range(i + 1, len(words)):
                    if offsets[i] + offsets[j] == len(tape):
                        if j - i >= 2:
                            forbidden = True
                        elif words[i] not in REPEATABLE_FUNCTION_WORDS:
                            forbidden = True
            if not forbidden:
                expected.add(words)
        actual = {tuple(record["words"]) for record in boundary_chart(slots)["records"]}
        assert actual == expected


def test_reparse_requires_whole_typed_agreement_valency_and_attachment():
    assert typed_parse(tuple("careful writers edit detailed reports".split())) == []  # No such combined layout is installed.
    assert typed_parse(tuple("writers carefully edit reports".split()))
    assert typed_parse(tuple("in offices writers edit reports".split()))
    assert typed_parse(tuple("writers edit reports from editors".split()))
    assert typed_parse(tuple("writer edit reports".split())) == []
    assert typed_parse(tuple("writers edits reports".split())) == []
    assert typed_parse(tuple("writers eat reports".split())) == []
    assert typed_parse(tuple("writers edit reports in".split())) == []
    assert typed_parse(tuple("writers edit reports from factories".split())) == []
    [parse] = typed_parse(tuple("careful writers who revise detailed reports in quiet offices silently edit brief essays from tired authors in large studios".split()))
    assert parse["relative_event"]["subject_shared_with_main_clause"]
    assert parse["attachments"] == ["event_location", "object_origin"]


def test_each_installed_frame_and_layout_has_independent_typed_parse():
    for frame in FRAMES:
        inventory = domains(frame)
        for roles in LAYOUTS.values():
            assert typed_parse(tuple(inventory[role][0] for role in roles))


def test_only_exact_centrally_admitted_survivors_are_rendered():
    result = run(min_letters=1)
    assert result["states_exhausted"]
    assert result["lexical_realizations"] > 10_000
    assert any(row["length_range"][1] >= 100 for row in result["rows"])
    assert all("records" not in row for row in result["rows"])
    for survivor in result["survivors"]:
        assert all(survivor["mechanical_checks"].values())
        assert survivor["normalized"] == survivor["normalized"][::-1]
        assert survivor["typed_parses"]
        assert survivor["reader_status"] == "unreviewed"
