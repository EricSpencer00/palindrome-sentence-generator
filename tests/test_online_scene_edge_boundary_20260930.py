import json
from pathlib import Path

from experiments.online_scene_edge_boundary_20260930 import run


def test_online_edge_artifact_is_reproducible_and_reports_residual():
    result = run()
    assert result["method"].startswith("online typed")
    assert result["candidate_count"] == 0
    assert result["novelty_preflight"]["not_complete_sentence_cartesian_sweep"]
    assert result["status"].startswith("no exact")
    saved = json.loads((Path(__file__).parents[1] / "runs" / "online-scene-edge-boundary-20260930.json").read_text())
    assert saved["attempts"] == result["attempts"]
