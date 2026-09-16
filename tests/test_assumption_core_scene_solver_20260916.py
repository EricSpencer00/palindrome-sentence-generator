import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.assumption_core_scene_solver_20260916 import run, tape


def test_bounded_run_has_two_audits_and_repairs_for_failures():
    result = run(48)
    assert result["tested"] == 48
    assert result["closures"] == []
    assert all("audit_a" in row and "audit_b" in row for row in result["records"])
    assert all(row.get("heldout_repair") for row in result["records"])


def test_exact_audit_is_independent_of_recorded_check():
    result = run(12)
    for row in result["records"]:
        assert row["audit_b"] == (tape(row["text"]) == tape(row["text"])[::-1])
