"""Tests for the post-nine natural ``extra help`` boundary repair."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "natural_location_extra_help_replay_20260913",
    ROOT / "experiments/inverted_question_natural_location_extra_help_replay_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_real_emitter_replays_ten_pairs_and_captures_next_mismatch():
    row = MODULE.replay_bridge(MODULE.ExtraHelpGrammar(6))
    assert row["ten_pairs_replayed"] and row["boundary_crossed"]
    assert row["bridge_stream"] == row["terminal_reverse_stream"] == "tatempleha"
    mismatch = row["first_post_nine_mismatch"]
    assert (mismatch["left_word"], mismatch["left_character"]) == ("hall", "l")
    assert (mismatch["right_word"], mismatch["right_character"]) == ("extra", "r")
    assert mismatch["emitter_rejected"]
    assert row["independent_parse"] and not row["exact_audit"]["exact"]


def test_extra_help_is_independently_parsed_and_typed():
    grammar = MODULE.ExtraHelpGrammar(1)
    text = "was it a temple hall that the extra help met at i saw"
    tree = MODULE.QUESTION.parse_tree(grammar, text)
    witness = MODULE.NATURAL.feature_witness(tree)
    assert tree is not None and witness["agreement_ok"] and witness["valency_ok"]
    assert any(role.get("type") == "person_group" for role in witness["semantic_roles"])


def test_long_control_is_complete_independently_reparsed_and_nonrepetitive():
    grammar = MODULE.ExtraHelpGrammar(6)
    state = MODULE.NATURAL.SECOND.SECOND.LOCATIVE.explicit_control(grammar)
    text = MODULE.BASE.render(state)
    row = MODULE.NATURAL.audit(grammar, text, "control", state.trace)
    content = [leaf.word for leaf in MODULE.BASE.ordered_leaves(state)
               if leaf.label not in MODULE.FIXED_LABELS]
    assert len(text.replace(" ", "")) >= 100
    assert row["independent_parse"] and row["feature_witness"]["agreement_ok"]
    assert row["feature_witness"]["valency_ok"]
    assert len(content) == len(set(content))


def test_extra_is_rejected_when_not_matching_live_residual():
    """The new lexical form is in the branch inventory, not an unfiltered escape."""
    state = MODULE.BASE.State((0,), (), (), "r", 1, 0, ())
    leaf = MODULE.BASE.Leaf(99, 99, "extra", "adj", 0, 0)
    assert not MODULE.lexical_unit_accepts(state, -1, [leaf])
