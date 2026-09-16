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


def test_novelty_preflight_blocks_before_model_load():
    preflight = novelty_preflight()
    assert preflight["performed_before_model_load"]
    assert preflight["blocked"]
    overlap_ids = {row["id"] for row in preflight["overlaps"]}
    assert {"gpt2-topic-half-decoder-20260915", "bpe-dual-continuation"} <= overlap_ids


def test_blocked_run_preserves_pivot_schema_and_registry_artifact():
    payload = run()
    assert payload["experiment_id"] == EXPERIMENT_ID
    assert payload["signature"] == SIGNATURE
    assert payload["pivot"]["status"] == "preflight_blocked"
    assert payload["stats"]["model_loaded"] == 0
    assert payload["stats"]["live_expansions"] == 0
    assert payload["rendered_candidates"] == []
    assert payload["rendered_probes"] == []
    assert payload["repair"]["operator"]
    assert payload["provenance"]["known_palindromes_used"] is False
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "docs/experiment-novelty-registry.json").read_text())
    row = next(item for item in registry["excluded"] if item["id"] == EXPERIMENT_ID)
    assert row["signature"] == SIGNATURE
    assert (root / row["artifact"]).exists()
    assert all((root / item).exists() for item in row["run_artifacts"])
    assert EVIDENCE == root / row["run_artifacts"][0]
