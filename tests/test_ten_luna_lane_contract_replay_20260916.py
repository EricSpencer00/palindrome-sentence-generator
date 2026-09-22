import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from experiments.validate_ten_luna_lane_contract_20260916 import main


def test_all_ten_orthogonal_lanes_have_independent_replay_and_repairs():
    report = main()
    assert report["status"] == "contract_validated_diagnostic_only"
    assert report["lane_count"] == 10
    assert report["exact_count"] == 0
    assert report["mechanically_admitted_count"] == 0
    assert all(row["rendered"].strip() for row in report["rows"])
    assert all(row["provenance_present"] for row in report["rows"])
    assert all(row["novelty_preflight_present"] for row in report["rows"])
    assert all(row["next_repair_present"] for row in report["rows"])


def test_lane_eight_uses_distinct_repair_not_canonical_repetition():
    report = main()
    lane = next(row for row in report["rows"] if row["lane"] == 8)
    assert "a man a plan a canal panama" not in lane["rendered"].lower()
    assert len(lane["rendered"]) > 100
