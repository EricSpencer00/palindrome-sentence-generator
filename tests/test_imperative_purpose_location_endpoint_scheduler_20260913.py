"""Tests for the imperative-purpose endpoint scheduler."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.imperative_purpose_location_endpoint_scheduler_20260913 import (
    FRAMES, endpoint_index, independent_parse, outside_in_ledger, run,
)


def test_scheduler_only_admits_three_pair_endpoint_joins():
    algebra = endpoint_index()
    assert algebra["joins"]
    assert all(row["matched_pairs"] >= 3 for row in algebra["joins"])
    assert all(len(row["chosen_surface_words"]) == 13 for row in algebra["joins"])


def test_independent_parser_rejects_valency_or_attachment_mutation():
    frame = next(frame for frame in FRAMES if frame.action == "repair" and frame.object == "canvas")
    text = " ".join(frame.words).capitalize() + "."
    assert independent_parse(text)["ok"]
    assert not independent_parse(text.replace("in", "at", 1))["ok"]
    assert not independent_parse(text.replace(frame.purpose_object, "model"))["ok"]


def test_ledger_rejects_mismatch_and_reports_first_pair():
    ledger = outside_in_ledger("Repair a clean model in a quiet studio to save a final paper.")
    assert not ledger["exact"] and ledger["first_mismatch"]["pair"] >= 1
    assert ledger["first_mismatch"]["left"] != ledger["first_mismatch"]["right"]


def test_fresh_run_preserves_per_channel_coverage_and_no_readability_claim():
    result = run(max_paths_per_channel=100_000)
    assert result["config"]["endpoint_index_before_ledger"]
    assert result["config"]["minimum_endpoint_matched_pairs"] == 3
    assert sum(row["scheduled_paths"] for row in result["channel_coverage"]) == len(result["records"])
    assert result["reader_facing_next_test"].startswith("Only an admitted exact surface")
    assert all(row["outside_in_ledger"]["exact"] for row in result["exact_candidates"])
    assert all(row["mechanically_admitted"] for row in result["admitted_candidates"])
