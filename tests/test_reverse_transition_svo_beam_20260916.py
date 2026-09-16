import json
from pathlib import Path

from experiments.reverse_transition_svo_beam_20260916 import (
    EVIDENCE,
    EXPERIMENT_ID,
    SIGNATURE,
    run,
)


def test_transition_beam_keeps_full_svo_probes_and_exact_join_fail_closed():
    payload = run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["signature"] == SIGNATURE
    registry = json.loads((Path(__file__).resolve().parents[1] / "docs/experiment-novelty-registry.json").read_text())
    assert payload["novelty_preflight"]["registry_entries"] == len(registry["entries"]) - 1
    assert payload["novelty_preflight"]["exact_signature_collision"] is False
    assert payload["stats"]["left_bank"] == 256
    assert payload["stats"]["right_bank"] == 256
    assert payload["stats"]["rendered_probes"] == 3
    assert payload["stats"]["exact"] == 0
    assert payload["stats"]["mechanically_admitted"] == 0
    assert all(row["letters"] >= 39 for row in payload["rendered_probes"])
    assert all(not row["exact"] for row in payload["rendered_probes"])
    assert payload["repair"]["status"] == "not_run"


def test_transition_route_is_registered_with_frozen_run():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    row = next(item for item in registry["entries"] if item["id"] == EXPERIMENT_ID)
    assert row["signature"] == SIGNATURE
    assert (root / row["artifact"]).exists()
    assert all((root / artifact).exists() for artifact in row["run_artifacts"])
    assert EVIDENCE == root / row["run_artifacts"][0]
