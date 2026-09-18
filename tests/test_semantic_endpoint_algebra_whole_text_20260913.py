"""Tests for endpoint-indexed whole-text construction."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.whole_text_palindrome_product_20260913 import compile_slots, construct, exhaustive_reference
from experiments.semantic_endpoint_algebra_whole_text_20260913 import (
    DERIVATIONS, endpoint_algebra, endpoint_realizations, independent_parse, render, run,
)


def test_endpoint_algebra_requires_two_real_pairs_and_preserves_roles():
    algebra = endpoint_algebra()
    assert algebra["minimum_matched_pairs"] == 2
    assert algebra["joins"]
    assert all(row["matched_pairs"] >= 2 for row in algebra["joins"])
    assert all(row["semantic_roles"][0] in {"agent_action", "subject_np"} for row in algebra["joins"])
    assert all(row["semantic_roles"][1] in {"locative_complement", "object_np"} for row in algebra["joins"])
    assert len(algebra["eligible_derivation_ids"]) > 0


def test_every_eligible_derivation_is_independently_grammatical():
    eligible = set(endpoint_algebra()["eligible_derivation_ids"])
    for derivation in DERIVATIONS:
        if derivation.identifier not in eligible:
            continue
        words = tuple(slot[0].form for slot in derivation.choices)
        parsed = independent_parse(derivation, render(words))
        assert parsed["ok"], (derivation.identifier, parsed)


def test_independent_tiny_oracle_still_matches_kernel_closures():
    slots = (("ij", "ix"), ("k",), ("ji", "zz"))
    result = construct(compile_slots(slots), max_states=1_000)
    assert {tuple(row["words"]) for row in result["records"]} == exhaustive_reference(slots) == {("ij", "k", "ji")}
    assert result["states_exhausted"] and not result["truncated"]


def test_fresh_endpoint_run_reports_true_scope_and_replay_requirements():
    result = run(max_states=1_000)
    assert result["config"]["endpoint_algebra_before_kernel"]
    assert result["config"]["minimum_endpoint_matched_pairs"] == 2
    assert result["config"]["single_intact_clause"]
    assert result["reader_facing_next_test"].startswith("Only an admitted closure")
    for row in result["exact_closures"]:
        assert row["endpoint_join"]["matched_pairs"] >= 2
        assert row["independent_terminal_path_replay"]["ok"]
        assert row["independent_parse"]["ok"]
        assert row["independent_exact_audit"]["exact"]

