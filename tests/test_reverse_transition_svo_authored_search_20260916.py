import json
from pathlib import Path

from experiments.reverse_transition_svo_authored_search_20260916 import EVIDENCE, EXPERIMENT_ID, SIGNATURE, run


def test_authored_search_uses_complete_grammar_and_exact_boundary_walker():
    payload = run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["signature"] == SIGNATURE
    assert payload["config"]["grammar"] == "DET SUBJ VERB DET ADJ OBJ PREP DET NOUN"
    assert payload["novelty_preflight"]["exact_signature_collision"] is False
    assert payload["stats"]["states"] > 0
    assert payload["stats"]["exact"] == 0
    assert payload["stats"]["mechanically_admitted"] == 0
    assert all(row["letters"] >= 39 for row in payload["rendered_probes"])


def test_authored_search_is_registered_as_same_family_repair():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    row = next(item for item in registry["entries"] if item["id"] == "reverse-transition-svo-beam-20260916")
    assert "experiments/reverse_transition_svo_authored_search_20260916.py" in row["repair_artifacts"]
    assert "runs/reverse-transition-svo-authored-search-20260916.json" in row["run_artifacts"]
    assert EVIDENCE.exists()
