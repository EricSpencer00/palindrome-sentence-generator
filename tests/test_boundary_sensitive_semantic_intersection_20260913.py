from pathlib import Path

import pytest

from experiments.acoustic_source_grammar_20260913 import SourceState, source_transitions, source_signature
from experiments.acoustic_target_grammar_20260913 import TargetState, target_transitions, target_signature
from experiments.acoustic_whole_sentence_validator_20260913 import parse_acoustic_sentence
from experiments.boundary_sensitive_semantic_intersection_20260913 import run, semantic_intersection
import experiments.boundary_sensitive_semantic_intersection_20260913 as constructor
from experiments.boundary_shifting_sentence_lattice_20260913 import search_lattice, toy_lattice
from llm_palindrome.admission import has_self_palindromic_proper_multiword_span, is_lexical_word


@pytest.fixture(scope="module")
def production():
    return run()


def test_production_crossing_is_certified_after_five_actual_pairs(production):
    assert production["stats"]["actual_endpoint_eligible_states"] == 18
    assert production["deepest"]["depth"] == 5
    for witness in production["production_relexicalization_witnesses"]:
        assert witness["production_not_fixture"]
        assert witness["actual_outer_pairs"] == 5
        assert witness["matched_tape"] == "timek"
        assert witness["provisional_complete_words"] == ("time",)
        assert witness["provisional_unfinished_word"] == "k"
        assert witness["final_unfinished_word"] == "timek"
        assert witness["crossed_provisional_cuts"] == (4,)
        assert not witness["final_boundaries"]
        whole = witness["whole_path_completion"]
        assert "".join(whole["source_tokens"]) == "".join(whole["target_tokens"]) == whole["normalized"]
        assert whole["normalized"][:5] == whole["normalized"][-5:][::-1] == "timek"
        assert whole["source_tokens"][:2] == ["time", "keepers"]
        assert whole["target_tokens"][0] == "timekeepers"
        assert all(is_lexical_word(word) for word in whole["target_tokens"])
        assert any(row["semantic_relation_valid"] for row in whole["independent_target_semantics"])
        assert not whole["exact_palindrome"]


def test_complete_production_witness_belongs_to_both_independent_languages(production):
    for witness in production["production_relexicalization_witnesses"]:
        whole = witness["whole_path_completion"]
        states = {SourceState("start")}
        for word in whole["source_tokens"]:
            states = {target for state in states for token, target in source_transitions(state) if token == word}
        source_keys = {source_signature(state) for state in states if source_signature(state) is not None}
        states = {TargetState("initial")}
        for word in whole["target_tokens"]:
            states = {target for state in states for token, target in target_transitions(state) if token == word}
        target_keys = {target_signature(state) for state in states if target_signature(state) is not None}
        assert source_keys and source_keys == target_keys


def test_source_target_and_final_parser_are_separate_declarations():
    for filename in ("acoustic_source_grammar_20260913.py", "acoustic_target_grammar_20260913.py", "acoustic_whole_sentence_validator_20260913.py"):
        source = Path("experiments", filename).read_text()
        assert "from experiments" not in source
        assert "import acoustic" not in source
    assert SourceState is not TargetState
    assert SourceState("start") != TargetState("initial")


@pytest.mark.parametrize("words", [
    "timekeepers record durations of calls that elk emit",
    "gamekeepers who use digital recorders carefully measure lengths of sounds which distant wolves produce",
    "beekeepers who carry portable recorders from local laboratories patiently log precise durations of faint noises that wild crickets make",
])
def test_independent_whole_sentence_event_binding(words):
    row, = parse_acoustic_sentence(words.split())
    assert row["semantic_relation_valid"]
    assert row["measurement"]["event_id"] == row["event"]["id"]
    assert row["measurement"]["unit"] == "seconds"


def test_independent_dimension_and_grammar_rejections():
    row, = parse_acoustic_sentence("timekeepers record weights of calls that elk emit".split())
    assert not row["semantic_relation_valid"]
    for text in ("time keepers record durations of calls that elk emit", "timekeepers records durations of calls that elk emit",
                 "timekeepers record durations of calls that elk emit frogs", "timekeepers record durations of calls that elk",
                 "unknownkeepers record durations of calls that elk emit"):
        assert not parse_acoustic_sentence(text.split())


def intersect_fixture(source_sentences, target_sentences):
    return semantic_intersection(toy_lattice(source_sentences), toy_lattice(target_sentences),
                                 lambda state: "fixture-event", lambda state: "fixture-event", probe_pairs=5)


@pytest.mark.parametrize("middle,kind,tape", [
    ("fg", "odd", "abcdefgfedcba"), ("fgg", "even", "abcdefggfedcba"),
])
def test_free_even_and_odd_centers_in_intersection(middle, kind, tape):
    source = ("a", "bcde", middle, "fedcba")
    target = ("abcde", middle, "fedcba")
    lattice, _ = intersect_fixture([source], [target])
    result = search_lattice(lattice, probe_pairs=5)
    row, = result["records"]
    assert row["tape"] == tape
    assert row["center_kind"] == kind
    assert row["matched_depth"] >= 5
    assert not has_self_palindromic_proper_multiword_span(row["words"])
    assert all(word != word[::-1] for word in row["words"])


@pytest.mark.parametrize("middle,code", [
    (("fg", "gf"), "proper_island_prunes"), (("level",), "single_or_multiword_center_prunes"),
])
def test_target_multiword_and_content_islands_are_pruned_online(middle, code):
    lattice, _ = intersect_fixture([("a", "bcde") + middle + ("edcba",)], [("abcde",) + middle + ("edcba",)])
    result = search_lattice(lattice, probe_pairs=5)
    assert not result["records"]
    assert result["stats"][code] == 1


def test_finite_intersection_accounting_matches_full_small_enumeration():
    source = [("a", "bcde", "fg", "fedcba"), ("a", "bcde", "fgg", "fedcba"),
              ("a", "bcde", "fg", "gf", "edcba"), ("hijkl", "mn", "mlkjih")]
    target = [("abcde", "fg", "fedcba"), ("abcde", "fgg", "fedcba"),
              ("abcde", "fg", "gf", "edcba"), ("hijkl", "mn", "mlkjih")]
    source_lattice, target_lattice = toy_lattice(source), toy_lattice(target)
    lattice, accounting = semantic_intersection(source_lattice, target_lattice, lambda s: "fixture-event", lambda s: "fixture-event")
    unrestricted, _ = semantic_intersection(source_lattice, target_lattice, lambda s: "fixture-event", lambda s: "fixture-event", require_shift=False)
    # The even-center and island surfaces share a tape.  Their two-by-two
    # source/target analyses count as four synchronized lexical path pairs,
    # not two unique target surfaces.
    expected_pairs = [(a, b) for a in source for b in target if "".join(a) == "".join(b)]
    assert len(expected_pairs) == 6
    def enumerate_actual_pairs(graph):
        found, pending = [], [graph.start]
        while pending:
            node = pending.pop()
            if node in graph.accepting:
                state = graph.nodes[node].parser_state
                found.append((source_lattice.nodes[state.source_node].parser_state,
                              target_lattice.nodes[state.target_node].parser_state))
            pending.extend(edge.target for edge in graph.edges if edge.source == node)
        return found
    actual_pairs = enumerate_actual_pairs(unrestricted)
    shifted_pairs = enumerate_actual_pairs(lattice)
    assert sorted(actual_pairs) == sorted(expected_pairs)
    assert len(shifted_pairs) == 5
    assert accounting["unrestricted_intersection"]["accepted_source_target_derivation_pairs"] == len(actual_pairs) == 6
    assert accounting["unrestricted_intersection"]["distinct_rendered_target_surfaces"] == len({b for a, b in actual_pairs}) == 4
    assert accounting["shift_filtered_intersection"]["accepted_source_target_derivation_pairs"] == len(shifted_pairs) == 5
    assert accounting["shift_filtered_intersection"]["distinct_rendered_target_surfaces"] == len({b for a, b in shifted_pairs}) == 3
    result = search_lattice(lattice, probe_pairs=5)
    expected = {sentence for sentence in target[:3] if not has_self_palindromic_proper_multiword_span(sentence)}
    assert {row["words"] for row in result["records"]} == expected
    assert len(result["records"]) == 3  # two analyses of the admissible even surface
    assert result["states_exhausted"]
    assert sum(result["state_depth_distribution"].values()) == result["stats"]["states"]


def test_semantically_incompatible_full_languages_have_empty_intersection():
    source = toy_lattice([("a", "bcde", "fg", "fedcba")])
    target = toy_lattice([("abcde", "fg", "fedcba")])
    lattice, counts = semantic_intersection(source, target, lambda state: "seconds", lambda state: "kilograms")
    assert counts["unrestricted_intersection"]["accepted_source_target_derivation_pairs"] == 0
    assert counts["unrestricted_intersection"]["distinct_rendered_target_surfaces"] == 0
    assert not search_lattice(lattice)["records"]


def test_production_exhaustion_and_next_mismatch_are_reported(production):
    assert production["construction"]["unrestricted_intersection"]["accepted_source_target_derivation_pairs"] == 8817984
    assert production["construction"]["unrestricted_intersection"]["distinct_rendered_target_surfaces"] == 8817984
    assert production["construction"]["shift_filtered_intersection"]["accepted_source_target_derivation_pairs"] == 4408992
    assert production["construction"]["shift_filtered_intersection"]["distinct_rendered_target_surfaces"] == 4408992
    assert production["states_exhausted"]
    assert sum(production["state_depth_distribution"].values()) == production["stats"]["states"] == 585
    assert production["next_endpoint_mismatch"]["next_pair"] == 6
    assert production["next_endpoint_mismatch"]["left_next_letters"] == ["e"]
    assert production["next_endpoint_mismatch"]["right_next_letters_read_inward"] == ["l"]
    assert production["exact_closures"] == production["unique_exact_surfaces"] == production["exact_rejected_surfaces"] == 0
    assert production["closure_reviews"] == production["pending_external_review"] == production["promoted_candidates"] == []


def test_every_rejected_closure_keeps_rendering_and_all_central_checks(monkeypatch):
    # Inject mechanism fixtures only into this audit test, never production.
    records = []
    for middle in ("fg", "fgg"):
        words = ("abcde", middle, "fedcba")
        tape = "".join(words)
        records.append({"words": words, "tape": tape, "letters": len(tape), "matched_depth": len(tape) // 2})
    monkeypatch.setattr(constructor, "search_lattice", lambda lattice, probe_pairs: {
        "channels": [], "records": records, "deepest": {"depth": -1}, "states_exhausted": True})
    report = constructor.run(min_letters=1, max_letters=240)
    assert report["exact_closures"] == report["unique_exact_surfaces"] == report["exact_rejected_surfaces"] == 2
    assert report["independent_final_parse_calls"] == 2
    for row in report["closure_reviews"]:
        assert row["rendered_diagnostic"]
        assert row["exact"] and row["central_admission"]["exact_letter_palindrome"]
        assert not row["central_admission"]["lexicon_words"]
        assert not row["candidate_accepted"]
        assert "independent_whole_sentence_semantics" in row["failures"]
        assert "production_relexicalization_at_five_actual_pairs" in row["failures"]
        assert row["external_provenance"] == "not checked"
    assert report["pending_external_review"] == report["promoted_candidates"] == []
