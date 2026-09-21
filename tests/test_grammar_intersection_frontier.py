import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("g", Path(__file__).parents[1] / "experiments/grammar_intersection_frontier_20260921.py")
g = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(g)

def test_audit_detects_non_palindrome_and_sha_direction():
    result = g.audit("a quiet gardener")
    assert not result["two_pointer_exact"]
    assert result["sha_equal"] is False

def test_frontier_is_independent_from_exact_audit():
    f = g.frontier("a quiet gardener", "gardener quiet a")
    assert f["matched_frontier_pairs"] >= 1
    assert "left_obligation" in f and "right_obligation" in f

def test_run_has_concrete_repair_and_provenance(tmp_path):
    g.main()
    data = g.RUN.read_text()
    assert "next_construction" in data
    assert "forward_sha256" in data and "reverse_sha256" in data
