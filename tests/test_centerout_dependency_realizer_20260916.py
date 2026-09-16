import json
from pathlib import Path

from experiments.centerout_dependency_realizer_20260916 import (
    EVIDENCE,
    EXPERIMENT_ID,
    PLANS,
    SIGNATURE,
    run,
)


def test_centerout_route_keeps_complete_typed_probes_and_zero_exact_rows():
    payload = run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["signature"] == SIGNATURE
    assert payload["novelty_preflight"]["registry_entries"] == 102
    assert payload["novelty_preflight"]["runtime_registry_entries"] == 103
    assert payload["novelty_preflight"]["exact_signature_collision"] is False
    assert payload["stats"]["plans"] == len(PLANS) == 3
    assert payload["stats"]["rendered_probes"] == 3
    assert payload["stats"]["exact"] == 0
    assert payload["stats"]["mechanically_admitted"] == 0
    assert all(row["topic_and_agreement"]["left"]["topic_continuity"] for row in payload["rendered_probes"])
    assert all(row["topic_and_agreement"]["right"]["argument_agreement"] for row in payload["rendered_probes"])
    assert all(not row["exact"] for row in payload["rendered_probes"])
    assert all(row["letters"] >= 39 for row in payload["rendered_probes"])


def test_centerout_route_is_registered_with_frozen_evidence():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    row = next(item for item in registry["entries"] if item["id"] == EXPERIMENT_ID)
    assert row["signature"] == SIGNATURE
    assert (root / row["artifact"]).exists()
    assert all((root / artifact).exists() for artifact in row["run_artifacts"])
    assert EVIDENCE == root / row["run_artifacts"][0]
