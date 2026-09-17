import json
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lane", ROOT / "experiments/luna_inflection_clitic_seam_scene_solver_20260917.py")
lane = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(lane)


def test_lane_emits_complete_prose_and_independent_audits():
    result = lane.run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["variants"] == 72
    assert result["stats"]["exact"] == 0
    assert all(";" in row["rendered"] and row["rendered"].endswith(".") for row in result["rendered_candidates"])
    assert all(row["audit"]["sha256_forward"] != row["audit"]["sha256_reverse"] for row in result["rendered_candidates"])
    assert all(not any(row["anti_shortcut_flags"].values()) for row in result["rendered_candidates"])
    assert result["failure_and_repair"]["next_repair"]


def test_persisted_run_has_provenance_and_actual_candidates():
    result = json.loads((ROOT / "runs/luna-inflection-clitic-seam-scene-solver-20260917.json").read_text())
    assert result["provenance"]["fresh_scene_authoring"]
    assert result["provenance"]["independent_audits"]
    assert result["rendered_candidates"]
