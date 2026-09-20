import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "bench"))
from language_gated_clause_intersection_20260920 import audit, load_edges  # noqa: E402


def test_transition_gate_keeps_anchor_edges_and_audits_exactness():
    edges = load_edges(Path("/nonexistent/count_2w.txt"))
    assert ("an", "aide") in edges
    assert ("inspire", "diana") in edges
    checked = audit("An aide rips nine memos; some men inspire Diana.")
    assert checked["pointer_exact"]
    assert checked["sha_equal"]
