"""Tests for cross-attachment residual repair and inside product."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.cross_attachment_inside_product_20260913 import (
    CORES, attachment_pairs,
    independent_parse, inside_product, render, run,
)


def test_cross_attachment_changes_original_core_and_reaches_six_pairs():
    pairs = attachment_pairs()
    deep = next(row for row in pairs if row["matched_pairs"] >= 6)
    assert deep["source_core"] != deep["selected_core"]
    assert deep["attachment_changed"]
    assert deep["matched_pairs"] >= 6


def test_selected_surface_is_independently_parsed():
    row = next(row for row in attachment_pairs() if row["matched_pairs"] >= 6)
    core = next(core for core in CORES if core.identifier == row["selected_core"])
    assert independent_parse(render(core.words))["ok"]


def test_inside_product_is_seeded_at_six_not_three():
    row = next(row for row in attachment_pairs() if row["matched_pairs"] >= 6)
    core = next(core for core in CORES if core.identifier == row["selected_core"])
    from experiments.whole_text_palindrome_product_20260913 import compile_slots
    product = inside_product(compile_slots(tuple((word,) for word in core.words)), core.words, row["matched_pairs"], max_states=100)
    assert product["seed_pairs"] >= 6
    assert product["states"] >= 1
    assert product["states_exhausted"] or product["truncated"]


def test_fresh_run_reports_cross_attachment_and_exact_candidates_only():
    result = run(max_states=100_000)
    assert result["config"]["cross_core_attachment_selection"]
    assert result["config"]["minimum_pairs_before_inside_product"] == 6
    assert result["attachment_pair_count"] > 0
    assert result["inner_product_runs"]
    assert all(row["outside_in_ledger"]["exact"] for row in result["exact_candidates"])
