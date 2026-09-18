"""Tests for residual-aware partial suffix scheduling."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.residual_aware_endpoint_expansion_20260913 import (
    CORES, endpoint_indices, independent_parse, outside_in_ledger, run, schedule_core,
)


def test_partial_suffix_index_records_real_three_pair_joins():
    algebra = endpoint_indices()
    assert algebra["joins"]
    assert all(row["matched_pairs"] >= 3 for row in algebra["joins"])
    assert any(":4:" in key for key in algebra["reverse_partial_suffix_index"])


def test_scheduler_carries_residual_and_never_enters_inner_before_six():
    algebra = endpoint_indices()
    core = next(core for core in CORES if core.identifier in algebra["eligible_core_ids"])
    row = schedule_core(core, algebra)
    assert row["endpoint_depth"] >= 3
    assert row["residual_expansion"]["required_depth"] == 6
    assert not row["residual_expansion"]["extended"]
    assert row["outside_in_ledger"]["first_mismatch"]
    deep = next(schedule_core(core, algebra) for core in CORES
                if core.identifier in algebra["eligible_core_ids"]
                and schedule_core(core, algebra)["endpoint_depth"] >= 6)
    assert deep["residual_expansion"]["extended"]
    assert deep["inner_expansion_ledger"]


def test_independent_parser_rejects_wrong_semantic_attachment():
    core = next(core for core in CORES if core.action == "draft")
    text = ("Draft a clean letter in a quiet office to write a final guide.")
    assert independent_parse(text)["ok"]
    assert not independent_parse(text.replace("guide", "model"))["ok"]


def test_run_reports_channel_coverage_and_no_readability_certification():
    result = run(max_paths_per_channel=100_000)
    assert result["config"]["residual_aware_expansion"]
    assert result["config"]["minimum_outer_pairs_before_inner_expansion"] == 6
    assert sum(row["scheduled_paths"] for row in result["channel_coverage"]) == len(result["records"])
    assert all(row["outside_in_ledger"]["exact"] for row in result["exact_candidates"])
    assert result["reader_facing_next_test"].startswith("Only an admitted exact surface")
