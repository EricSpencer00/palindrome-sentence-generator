"""Tests for the cross-word tail-segmentation experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.cross_word_tail_segmentation_product_20260913 import (
    CORES,
    endpoint_joins,
    independent_parse,
    render,
    run,
    tail_segmentation_index,
)


def test_reverse_index_contains_partial_word_and_crosses_a_tail_boundary():
    index = tail_segmentation_index()
    assert index["max_depth"] > 8
    assert 1 in index["depths"]
    joins = endpoint_joins()
    assert joins
    # Six matching characters cannot come from the first terminal word
    # alone: the reverse stream has crossed into the preceding tail word.
    deep = next(row for row in joins if row["matched_pairs"] >= 6)
    assert deep["cross_word_tail_segmentation"]
    assert deep["reversed_terminal_letters"][4] == deep["terminal_words"][-2][-1]
    assert deep["reversed_terminal_letters"].startswith(deep["opening_letters"][:6])


def test_composite_surface_is_one_typed_imperative_and_independently_reparsed():
    row = next(row for row in endpoint_joins() if row["matched_pairs"] >= 6)
    assert len(row["composite_words"]) == 13
    assert row["middle_words"] == row["composite_words"][4:9]
    parsed = independent_parse(render(row["composite_words"]))
    assert parsed["ok"]
    assert parsed["agreement_ok"] and parsed["valency_ok"]


def test_free_center_is_not_entered_before_eight_real_pairs():
    result = run(max_paths_per_channel=100_000, max_states=100_000)
    assert result["config"]["minimum_pairs_before_free_center"] == 8
    assert all(row["endpoint_join"]["matched_pairs"] < 8 for row in result["residual_diagnostics"])
    assert all(row["endpoint_join"]["matched_pairs"] >= 8 for row in result["free_center_runs"])
    assert all(row["outside_in_ledger"]["exact"] is False for row in result["residual_diagnostics"])
    # The current authored inventory has a finite six-pair maximum; this is
    # a transparent diagnostic, not a readability result.
    assert not result["free_center_runs"]
    assert not result["exact_candidates"]


def test_all_authored_cores_have_independent_surface_roles():
    # The composite parser is independent of Core fields, but this sanity
    # check ensures the authored role inventory itself remains well typed.
    core = next(core for core in CORES if independent_parse(render(core.words))["ok"])
    assert core.action and core.purpose and core.purpose_object
