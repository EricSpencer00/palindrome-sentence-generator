import json
from pathlib import Path

from experiments.semantic_relation_alignment_20260916 import (
    EVIDENCE,
    EXPERIMENT_ID,
    FRAME_EDGES,
    SIGNATURE,
    load_evidence,
    replay_ledger,
    run,
)


def test_packaged_failure_evidence_is_frozen_and_fail_closed():
    payload = load_evidence()
    assert payload == run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["signature"] == SIGNATURE
    assert payload["stats"]["frames"] == len(FRAME_EDGES) == 4
    assert payload["stats"]["exact"] == 0
    assert payload["stats"]["admitted"] == 0
    assert payload["stats"]["max_letters"] == 0
    assert payload["novelty_preflight"]["registry_entries"] == 94
    assert payload["novelty_preflight"]["exact_signature_collision"] is False


def test_ledger_replay_is_exact_character_check():
    assert replay_ledger("Able was I ere I saw Elba.")
    assert not replay_ledger("A readable sentence.")


def test_registry_points_to_existing_unique_artifact_and_run_record():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    matches = [row for row in registry["entries"] if row["id"] == EXPERIMENT_ID]
    assert len(matches) == 1
    row = matches[0]
    assert row["signature"] == SIGNATURE
    assert (root / row["artifact"]).exists()
    assert all((root / artifact).exists() for artifact in row["run_artifacts"])
    assert EVIDENCE == root / row["run_artifacts"][0]
