import json
from pathlib import Path

from experiments.reverse_transition_svo_adjunct_repair_20260916 import (
    BASE_EVIDENCE,
    EVIDENCE,
    EXPERIMENT_ID,
    SIGNATURE,
    run,
)


def test_repair_uses_measured_edge_and_complete_adjunct_grammar():
    payload = run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["repair_of"] == "reverse-transition-svo-beam-20260916"
    assert payload["novelty_preflight"]["exact_signature_collision"] is False
    assert payload["config"]["grammar"] == "DET SUBJ VERB DET ADJ OBJ PREP DET NOUN"
    assert payload["config"]["mismatch_keys"] == ["ab"]
    assert payload["config"]["held_out_terminals"] == ["area", "cinema", "opera", "plaza", "villa"]
    assert payload["stats"]["exact"] == 0
    assert payload["stats"]["mechanically_admitted"] == 0
    assert payload["stats"]["rendered_probes"] == 3
    assert all(row["letters"] >= 39 for row in payload["rendered_probes"])
    for row in payload["rendered_probes"]:
        tokens = row["rendered"].replace(".", "").split()
        for index, token in enumerate(tokens):
            if token in {"in", "on", "at", "by", "for", "near", "over", "under", "with"}:
                assert tokens[index + 1] in {"a", "an", "the", "our", "my", "one", "no"}
    assert all(not row["exact"] for row in payload["rendered_probes"])


def test_repair_is_registered_under_base_family_and_preserves_base_evidence():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    row = next(item for item in registry["entries"] if item["id"] == "reverse-transition-svo-beam-20260916")
    assert "experiments/reverse_transition_svo_adjunct_repair_20260916.py" in row["repair_artifacts"]
    assert "runs/reverse-transition-svo-beam-adjunct-repair-20260916.json" in row["run_artifacts"]
    assert BASE_EVIDENCE.exists()
    assert EVIDENCE.exists()
