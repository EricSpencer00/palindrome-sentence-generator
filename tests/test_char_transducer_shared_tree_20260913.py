"""Invariants for the single-tree character transducer experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "char_transducer_shared_tree_20260913",
    ROOT / "experiments/char_transducer_shared_tree_20260913.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_one_root_owns_every_surface_leaf():
    grammar = MODULE.FeatureGrammar()
    tree = grammar.expand(grammar.start())
    leaves = MODULE.slots(tree)
    assert tree.symbol.name == "S"
    assert len(leaves) == 24
    assert all(slot.symbol.name == "T" for slot in leaves)
    assert leaves[0].symbol.role == "subject_det"
    assert leaves[-1].symbol.role == "final_object"


def test_independent_reparse_checks_real_feature_roles():
    grammar = MODULE.FeatureGrammar()
    control = (
        "A patient curator who repairs a damaged artifact records a valuable collection "
        "after a quiet trial while a local archive monitors a detailed report."
    )
    assert MODULE.parse_tree(grammar, control) is not None
    assert MODULE.parse_tree(grammar, control.replace("monitors", "catalogs")) is None
    assert MODULE.parse_tree(grammar, control.replace("A patient", "An patient")) is None


def test_residual_transducer_requires_all_right_leaves_before_closure():
    grammar = MODULE.FeatureGrammar()
    stats = MODULE.Counter()
    rows = MODULE.joint_transduce(MODULE.slots(grammar.expand(grammar.start())), grammar, limit=10, state_limit=100, stats=stats)
    assert rows == []
    # The search attempts a right outer leaf immediately after left emission;
    # it cannot assign a complete left span before any cancellation attempt.
    assert stats["right_leaf_attempts_after_left"] > 0
    assert stats["left_char_emissions"] > 0
    assert stats["right_char_emissions"] > 0


def test_bounded_run_has_actual_controls_but_no_unparsed_fragment_witness():
    result = MODULE.run(assignment_limit=8, closure_limit=3)
    assert result["exact_closures"] == []
    assert result["config"]["one_connected_tree"]
    assert result["config"]["grammar_owns_every_leaf"]
    assert result["config"]["closure_requires_all_leaves"]
    assert result["config"]["alternating_outer_leaf_schedule"]
    assert result["stats"]["right_leaf_attempts_after_left"] > 0
    assert all(row["independent_parse"] for row in result["complete_grammar_controls"])
    assert all(row["record_kind"] == "complete_connected_grammar_control" for row in result["complete_grammar_controls"])


def test_exactness_is_independent_of_provenance():
    grammar = MODULE.FeatureGrammar()
    row = MODULE.audit(grammar, "Stressed desserts.", "untrusted_fragment", ("untrusted",))
    assert row["independent_exact_audit"]["exact"]
    assert not row["independent_parse"]
    assert not row["mechanically_admitted"]
    assert "independent_complete_reparse_failed" in row["rejection_codes"]


def test_shifted_word_boundaries_are_still_character_exact():
    audit = MODULE.exact_audit("ab ccb a")
    assert audit["exact"]
    assert audit["shifted_word_boundaries"]
    oracle = MODULE.frontier_schedule_oracle(("ab", "ccb", "a"))
    assert oracle["exact"]
    assert oracle["exact_audit"]["shifted_word_boundaries"]
    assert oracle["stats"]["residual_cancellations"] > 0
