"""Tests for the natural villa-attributed collective-help topology repair."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "natural_location_villa_help_replay_20260913",
    ROOT / "experiments/inverted_question_natural_location_villa_help_replay_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_real_emitter_crosses_former_conflict_and_reproduces_next_one():
    row = MODULE.replay_bridge(MODULE.VillaHelpGrammar(6))
    assert row["villa_boundary_crossed"]
    assert row["bridge_stream"] == row["terminal_reverse_stream"] == "tatemplehall"
    assert row["pairs_replayed"] == 12
    failure = row["first_post_repair_mismatch"]
    assert (failure["slot"], failure["word"], failure["character"]) == (
        "noun_subject_modifier", "villa", "i"
    )
    assert failure["emitter_rejected"]
    assert row["independent_parse"] and not row["exact_audit"]["exact"]


def test_villa_help_is_an_independently_parsed_typed_phrase():
    grammar = MODULE.VillaHelpGrammar(6)
    text = "was it a temple hall that the villa help met at i saw"
    tree = MODULE.QUESTION.parse_tree(grammar, text)
    witness = MODULE.NATURAL.feature_witness(tree)
    assert tree is not None and witness["agreement_ok"] and witness["valency_ok"]
    assert any(role.get("type") == "person_group" for role in witness["semantic_roles"])


def test_complete_recursive_control_is_long_and_independently_validated():
    grammar = MODULE.VillaHelpGrammar(6)
    state = MODULE.NATURAL.SECOND.SECOND.LOCATIVE.explicit_control(grammar)
    text = MODULE.BASE.render(state)
    row = MODULE.NATURAL.audit(grammar, text, "control", state.trace)
    content = [leaf.word for leaf in MODULE.BASE.ordered_leaves(state)
               if leaf.label not in MODULE.FIXED_LABELS]
    assert len(text.replace(" ", "")) >= 100
    assert row["independent_parse"] and row["feature_witness"]["agreement_ok"]
    assert row["feature_witness"]["valency_ok"]
    assert len(content) == len(set(content))
