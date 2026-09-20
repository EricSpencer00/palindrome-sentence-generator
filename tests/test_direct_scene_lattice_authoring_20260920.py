import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "experiments/direct_scene_lattice_authoring_20260920.py"
spec = importlib.util.spec_from_file_location("lane", P)
lane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lane)

def test_run_is_bounded_and_audited():
    result = lane.main()
    assert result["states"] == len(lane.LEFT) * len(lane.RIGHT) * len(lane.SEAMS)
    assert result["exact_candidates"] == []
    assert result["best"]["audit"]["sha_equal"] is False

def test_seed_like_exact_audit_independent():
    a = lane.audit("An aide rips nine memos; some men inspire Diana.")
    assert a["exact"] and a["sha_equal"] and a["letters"] == 38
