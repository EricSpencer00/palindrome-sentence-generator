import json
from pathlib import Path
from experiments.seed_joint_slot_resegment_20260916 import run, EXPERIMENT_ID

def test_joint_lane_has_fresh_complete_prose_and_independent_audits():
    p = run()
    assert p["experiment_id"] == EXPERIMENT_ID
    assert p["stats"]["joint_assignments"] == 9
    assert p["novelty_preflight"]["status"] == "passed"
    assert p["benchmark"]["used_as_output"] is False
    for row in p["candidates"]:
        assert row["rendered"].endswith(".")
        assert row["exact_audit"]["algorithm"].startswith("independent_two_pointer")
        assert row["exact_audit"]["sha_equal"] is False
        assert row["provenance"]["source_sentences_copied"] is False
        assert row["provenance"]["seed_used_as_output"] is False

def test_artifact_is_reproducible():
    path = Path(__file__).resolve().parents[1] / "runs/seed-joint-slot-resegment-20260916.json"
    payload = run(); path.write_text(json.dumps(payload, indent=2) + "\n")
    assert json.loads(path.read_text())["stats"] == payload["stats"]
