import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("lane", ROOT / "experiments" / "reverse_edge_resegmented_scene_20260917.py")
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)


def test_live_reverse_edges_emit_complete_scene_witnesses_and_debt():
    result = lane.run()
    assert result["stats"]["rendered"] == 0
    assert result["stats"]["exact"] == 0
    assert result["diagnostic_frontiers"]
    assert all(row["audit"]["exact"] for row in result["candidates"])


def test_reverse_edges_are_not_self_palindromic_shortcuts():
    assert all(a != b and a[::-1] != b for a, b in lane.REVERSE_EDGES.items())
    assert lane.live_edge_search(lane.SCENES[0])[1]
