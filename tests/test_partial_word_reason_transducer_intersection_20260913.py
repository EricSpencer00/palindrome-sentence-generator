"""Tests for the partial-word reason transducer."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/partial_word_reason_transducer_intersection_20260913.py"
spec = importlib.util.spec_from_file_location("partial_reason_transducer_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = MODULE
spec.loader.exec_module(MODULE)


def test_forward_and_reverse_tries_retain_inside_word_state():
    forward = MODULE.Trie(("bright",))
    reverse = MODULE.Trie(("bright",), reverse=True)
    node = forward.advance(0, "b")
    assert node is not None and forward.next_chars(node) == ("r",)
    reverse_node = reverse.advance(0, "t")
    assert reverse_node is not None and reverse.next_chars(reverse_node) == ("h",)
    reverse_node = reverse.advance(reverse_node, "h")
    assert reverse_node is not None and reverse.next_chars(reverse_node) == ("g",)


def test_grammar_generated_reason_surface_independently_parses_and_is_typed():
    words = MODULE.plan_words(MODULE.EVENTS[0])
    text = " ".join(words)
    row = MODULE.audit(text, "grammar_generated_smoke")
    assert words == ("the", "lamp", "glows", "because", "the", "room", "is", "bright")
    assert row["independent_parse"]
    assert row["feature_witness"]["discourse_function"] == "causal_explanation"
    assert row["feature_witness"]["agreement_ok"]
    assert row["feature_witness"]["valency_ok"]
    assert row["feature_witness"]["subject_action_ok"]
    assert row["central_admission"]["lexicon_words"]
    assert row["central_admission"]["distinct_words"]
    assert row["central_admission"]["no_self_palindromic_proper_multiword_span"]
    assert row["central_admission"]["not_forbidden_catalogue_endpoint_scaffold"]


def test_transducer_exposes_equal_pairs_then_first_real_trie_mismatch():
    result = MODULE.run(max_states=100000)
    assert result["config"]["partial_word_transducer"]
    assert result["config"]["forward_and_reverse_lexical_tries"]
    assert result["config"]["equal_character_pair_emission"]
    assert result["config"]["advance_role_only_on_word_completion"]
    assert result["config"]["preauthored_full_control"] is False
    assert result["states_exhausted"]
    assert result["stats"]["exact_closures"] == 0
    assert result["rendered_candidates"] == []
    first = result["first_viable_state"]
    assert first["length"] == 1
    assert first["pair_trace"] == [[1, "t", "t"]]
    deepest = result["deepest_state"]
    assert deepest["length"] == 2
    assert deepest["pair_trace"][-1] == [2, "h", "h"]
    assert deepest["attempts"]["matching_next"] == []
    assert result["first_rejection"]["rejection"]


def test_wrong_causal_property_fails_independent_parse():
    wrong = "the lamp glows because the room is blue"
    row = MODULE.audit(wrong, "tampered_reason")
    assert not row["independent_parse"]
    assert "independent_complete_reparse_failed" in row["rejection_codes"]
    assert "subject_action_semantics_failure" in row["rejection_codes"]

