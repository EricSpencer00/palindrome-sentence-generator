import json
from pathlib import Path

from experiments.scene_lattice_semantic_product_20260917 import OUT, run


def test_scene_lattice_artifact_is_reproducible_and_has_no_unverified_survivor():
    data = run()
    assert all(not scene["exact_candidates"] for scene in data["scenes"].values())
    assert all(scene["first_mismatch_frontiers"] for scene in data["scenes"].values())
    assert data["reader_gate"].startswith("closed")
    assert json.loads(OUT.read_text())["experiment_id"] == data["experiment_id"]
