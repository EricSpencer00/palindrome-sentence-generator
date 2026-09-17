from pathlib import Path
import json
from experiments.residual_slot_repair_20260917 import run, EXPERIMENT_ID
from llm_palindrome.admission import normalize_letters

def test_residual_repair_is_local_and_independently_validated():
    report = run()
    assert report["repair"]["operator"] == "residual-inflection-slot"
    assert report["stats"] == {"retained_seed_pairs": 1, "slot_assignments": 2, "exact_candidates": 0}
    assert report["seed"]["audit"]["first_residual"]
    for row in report["rendered_candidates"]:
        tape = normalize_letters(row["rendered"])
        assert row["audit"]["two_pointer_exact"] == (tape == tape[::-1])
        assert row["provenance"]["heldout_inflection_domain"]
        assert row["provenance"]["mirrored_units"] is False

def test_registry_and_run_artifact():
    assert Path("runs/residual-slot-repair-20260917.json").exists()
    registry = json.loads(Path("docs/experiment-novelty-registry.json").read_text())
    assert any(entry["id"] == EXPERIMENT_ID for entry in registry["entries"])
