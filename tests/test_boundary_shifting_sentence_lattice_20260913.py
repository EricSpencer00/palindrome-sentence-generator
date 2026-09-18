from pathlib import Path

import pytest

from experiments.art_conservation_sentence_validator_20260913 import parse_sentence
from experiments.boundary_shifting_sentence_lattice_20260913 import (
    ACTORS, ParseState, compile_lattice, lattice_statistics, search_lattice, toy_lattice, transitions,
)


def test_provisional_boundary_can_be_crossed_by_a_final_word():
    lattice = toy_lattice([("a", "b", "cd", "cba"), ("ab", "cd", "cba")])
    result = search_lattice(lattice, probe_pairs=2)
    final = next(row for row in result["records"] if row["words"] == ("ab", "cd", "cba"))
    assert final["tape"] == "abcdcba"
    assert final["center_kind"] == "odd"
    assert final["center_characters"] == 1
    assert 1 not in final["left_boundary_depths"]
    assert any(w["side"] == "left" and w["provisional_cut"] == 1 for w in final["provisional_boundary_crossings"])


def test_free_even_midpoint_inside_a_word():
    result = search_lattice(toy_lattice([("ab", "cdd", "cba")]), probe_pairs=2)
    row, = result["records"]
    assert row["tape"] == "abcddcba"
    assert row["center_kind"] == "even"
    assert row["matched_depth"] == 4


def test_online_multiword_island_prune():
    result = search_lattice(toy_lattice([("ab", "cd", "dc", "ba")]), probe_pairs=2)
    assert not result["records"]
    assert result["stats"]["proper_island_prunes"] == 1
    assert result["prune_witnesses"][0]["depth"] == 2
    assert result["prune_witnesses"][0]["remaining_word_range"] == [2, 2]


def test_function_word_center_is_not_falsely_pruned():
    result = search_lattice(toy_lattice([("ab", "a", "ba")]), probe_pairs=1)
    row, = result["records"]
    assert row["tape"] == "ababa"
    assert row["center_kind"] == "function_word_exception"


@pytest.mark.parametrize("words", [
    "traders restore faded red art",
    "printmakers carefully mend torn prints",
    "print makers mend torn prints",
    "wood carvers repair chipped wood cuts",
    "curators who consult detailed records from national archives patiently restore faded paintings in quiet galleries",
    "restorers repair damaged artwork",
    "restorers repair damaged art work",
])
def test_independent_complete_event_grammar(words):
    rows = parse_sentence(words.split())
    assert rows and all(row["semantic_relation_valid"] for row in rows)
    assert all(row["event_effect_witness"]["predicted"] < row["event_effect_witness"]["initial"] for row in rows)


@pytest.mark.parametrize("words", [
    "curators mend chipped sculptures", "curators restore torn sculptures",
    "curators damage damaged art",
])
def test_independent_counterfactual_semantics_rejects_invalid_events(words):
    assert not any(row["semantic_relation_valid"] for row in parse_sentence(words.split()))


def test_complete_parser_does_not_accept_a_fragment_or_unknown_lexeme():
    assert not parse_sentence("traders restore faded".split())
    assert not parse_sentence("traders restore faded quaz art".split())
    assert not parse_sentence("traders restore faded art because".split())


def test_live_nodes_retain_lexical_prefix_and_semantic_states():
    lattice = compile_lattice(ParseState("start"), transitions, lambda state: state.phase in {"patient_done", "done"})
    assert all(isinstance(node.parser_state, ParseState) for node in lattice.nodes)
    assert any(node.lexical_prefix == "print" and node.parser_state.phase == "start" for node in lattice.nodes)
    assert any(node.boundary and node.parser_state.phase == "print_actor" for node in lattice.nodes)
    assert any(node.parser_state.verb == "restore" and node.parser_state.defect == "faded" and not node.boundary for node in lattice.nodes)
    assert not set(ACTORS) & {"see", "go", "no"}


def test_independent_validator_does_not_import_generator():
    source = Path("experiments/art_conservation_sentence_validator_20260913.py").read_text()
    assert "from experiments" not in source
    assert "import boundary_shifting" not in source


def test_online_catalogue_endpoint_scaffold_exclusion():
    result = search_lattice(toy_lattice([("go", "abc", "cba", "dog")]), probe_pairs=1)
    assert not result["records"]
    # No exactness or successful midpoint is needed to exclude a scaffold.
    assert result["stats"]["catalogue_scaffold_prunes"] == 1


def test_finite_language_counts_variable_tokens_and_lengths():
    lattice = toy_lattice([("a", "b", "cd", "cba"), ("ab", "cd", "cba"), ("ab", "cdd", "cba")])
    assert lattice_statistics(lattice) == {"acyclic": True, "accepted_lexical_paths": 3,
                                          "letter_length_range": [7, 8], "word_count_range": [3, 4]}


def test_cyclic_grammar_fails_closed():
    lattice = compile_lattice(0, lambda state: [("word", state)], lambda state: True)
    with pytest.raises(ValueError, match="acyclic"):
        search_lattice(lattice)


def test_every_generator_lexical_transition_has_independently_valid_completion():
    """Edge coverage, not a misleading claim to enumerate the whole language."""
    initial = ParseState("start")
    prefixes = {initial: ()}
    pending = [initial]
    edges = []
    while pending:
        state = pending.pop()
        for word, target in transitions(state):
            edges.append((state, word, target))
            if target not in prefixes:
                prefixes[target] = prefixes[state] + (word,)
                pending.append(target)
    def completion(state):
        if state.phase in {"patient_done", "done"}:
            return ()
        word, target = transitions(state)[0]
        return (word,) + completion(target)
    for state, word, target in edges:
        sentence = prefixes[state] + (word,) + completion(target)
        parses = parse_sentence(sentence)
        assert parses and any(row["semantic_relation_valid"] for row in parses), sentence


def test_single_content_palindrome_is_not_a_legal_island_exception():
    result = search_lattice(toy_lattice([("ab", "level", "ba")]), probe_pairs=1)
    assert not result["records"]
    assert result["stats"]["single_or_multiword_center_prunes"] == 1
