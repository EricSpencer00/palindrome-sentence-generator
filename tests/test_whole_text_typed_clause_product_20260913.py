"""Tests for the separately compiled whole-text typed-clause product."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.whole_text_palindrome_product_20260913 import (
    compile_slots, construct, exhaustive_reference,
)
from experiments.whole_text_typed_clause_product_20260913 import (
    PLANS, grammar_for, independent_parse, render, run,
)


def test_tiny_exhaustive_oracle_matches_every_kernel_closure_with_shifted_boundaries():
    # The only exact derivation is deliberately split at different word
    # boundaries in the two alternatives; no sentence text is used as corpus
    # or candidate material.
    slots = (("ab", "ax"), ("c", "cba"), ("ba", "zz"))
    grammar = compile_slots(slots)
    expected = exhaustive_reference(slots)
    result = construct(grammar, max_states=10_000)
    observed = {tuple(row["words"]) for row in result["records"]}
    assert observed == expected == {("ab", "c", "ba")}
    assert result["states_exhausted"] and not result["truncated"]


def test_kernel_accepts_even_and_odd_centres_inside_or_across_words():
    odd = construct(compile_slots((("ab",), ("c",), ("ba",))), max_states=100)
    even = construct(compile_slots((("de",), ("ed",))), max_states=100)
    assert odd["records"][0]["exact"] and odd["records"][0]["center_characters"] == 1
    assert even["records"][0]["exact"] and even["records"][0]["center_characters"] == 0


def test_each_plan_has_independent_complete_typed_parse():
    for plan in PLANS:
        words = tuple(choices[0].form for choices in plan.words)
        parsed = independent_parse(plan, render(words))
        assert parsed["ok"], (plan.name, parsed)
        assert parsed["agreement_ok"] and parsed["valency_ok"]


def test_full_product_is_finite_and_does_not_promote_unreadability():
    result = run(max_states=2_000)
    assert result["config"]["separate_plan_compilation"]
    assert result["config"]["even_and_odd_center_meets"]
    assert result["reader_facing_next_test"].startswith("Only a mechanically admitted")
    for row in result["exact_closures"]:
        assert row["independent_parse"]["ok"]
        assert row["independent_exact_audit"]["exact"]
