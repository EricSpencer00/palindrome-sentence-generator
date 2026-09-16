import json
from pathlib import Path

from experiments.live_gpt2_character_decoder_preflight_20260916 import (
    EVIDENCE,
    EXPERIMENT_ID,
    SIGNATURE,
    independent_validation,
    novelty_preflight,
    run,
)


def test_independent_tape_hash_and_two_pointer_validation():
    exact = independent_validation("abc cba")
    assert exact["exact"]
    assert exact["two_pointer"]["exact"]
    assert len(exact["sha256"]) == 64
    nonexact = independent_validation("ordinary prose")
    assert not nonexact["exact"]
    assert nonexact["two_pointer"]["first_mismatch"] is not None


def test_revised_novelty_preflight_accepts_orthogonal_character_state():
    preflight = novelty_preflight()
    assert preflight["performed_before_model_load"]
    assert not preflight["blocked"]
    assert preflight["operator"].startswith("direct character equation")


def test_live_run_preserves_complete_prose_probe_and_registry_artifact():
    payload = run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["signature"] == SIGNATURE
    assert not payload["novelty_preflight"]["blocked"]
    assert payload["stats"]["model_loaded"] == 1
    assert payload["stats"]["live_expansions"] > 0
    assert payload["stats"]["exact"] == 0
    assert payload["stats"]["mechanically_admitted"] == 0
    assert payload["stats"]["rendered_probes"] >= 1
    assert payload["stats"]["final_states"] >= 1
    assert payload["stats"]["live_expansions"] > 0
    assert payload["rendered_probes"][0]["left_complete"]
    assert payload["rendered_probes"][0]["right_complete"]
    assert payload["rendered_probes"][0]["letters"] > 38
    assert payload["rendered_probes"][0]["live_character_transitions"] > 0
    assert payload["rendered_probes"][0].get("control_probe") is not True
    assert payload["rendered_probes"][0]["provenance"]["fixed_tape"] is False
    assert payload["rendered_probes"][0]["provenance"]["reverse_emission"] is False
    assert payload["repair"]["operator"]
    assert payload["provenance"]["known_palindromes_used"] is False
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    row = next(item for item in registry["entries"] if item["id"] == EXPERIMENT_ID)
    assert row["signature"] == SIGNATURE
    assert (root / row["artifact"]).exists()
    assert all((root / item).exists() for item in row["run_artifacts"])
    assert EVIDENCE == root / row["run_artifacts"][0]
