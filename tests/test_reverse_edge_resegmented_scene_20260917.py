import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("lane", ROOT / "experiments" / "reverse_edge_resegmented_scene_20260917.py")
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)


def test_live_reverse_edges_emit_complete_scene_witnesses_and_debt():
    result = lane.run()
    assert result["stats"]["rendered"] == 3
    assert result["stats"]["exact"] == 0
    for row in result["candidates"]:
        assert row["semantic_graph"]["edge_trace"]
        assert row["live_obligation"]["cross_boundary_resegmentation"]
        assert row["live_obligation"]["remaining_debt"]
        assert row["audit"]["independent_pointer_exact"] is False
        assert row["provenance"]["whole_tape_reversed"] is False


def test_reverse_edges_are_not_self_palindromic_shortcuts():
    assert all(a != b and a[::-1] != b for a, b in lane.REVERSE_EDGES.items())
