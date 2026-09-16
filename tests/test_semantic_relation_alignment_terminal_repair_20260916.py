import json
from pathlib import Path

from experiments.semantic_relation_alignment_terminal_repair_20260916 import (
    EVIDENCE,
    EXPERIMENT_ID,
    SIGNATURE,
    run,
)


def test_terminal_repair_is_same_family_and_has_no_exact_closure():
    payload = run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["repair_of"] == "semantic-relation-alignment-20260916"
    assert payload["signature"].startswith("semantic-relation-alignment|")
    assert payload["signature"].endswith("terminal-compatible-phrase-spans|bounded-odd-center")
    assert payload["novelty_preflight"]["registry_entries"] == 95
    assert payload["novelty_preflight"]["exact_signature_collision"] is False
    assert payload["stats"]["exact"] == 0
    assert payload["stats"]["admitted"] == 0
    assert payload["config"]["terminal_span_count"] > 1
    assert payload["config"]["odd_center"]


def test_repair_run_is_registered_as_base_family_artifact():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    row = next(item for item in registry["entries"] if item["id"] == "semantic-relation-alignment-20260916")
    assert "experiments/semantic_relation_alignment_terminal_repair_20260916.py" in row["repair_artifacts"]
    assert "runs/semantic-relation-alignment-terminal-repair-20260916.json" in row["run_artifacts"]
    assert EVIDENCE.exists()
    assert EVIDENCE == root / "runs/semantic-relation-alignment-terminal-repair-20260916.json"
