import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "seed_benchmark_live_semantic_slot_expansion_20260916",
    ROOT / "experiments/seed_benchmark_live_semantic_slot_expansion_20260916.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_fresh_scene_is_longer_than_benchmark_and_seed_free():
    data = MODULE.run()
    row = data["candidates"][0]
    assert row["letters"] > 38
    assert row["seed_benchmark"]["letters"] == 38
    assert row["seed_benchmark"]["used_in_output"] is False
    assert row["prose_checks"]["complete_clauses"]
    assert row["choices_selected_before_rendering"] is True


def test_live_slot_equation_and_independent_hashes_are_recorded():
    data = MODULE.run()
    row = data["candidates"][0]
    assert len(row["live_bilateral_equation"]) == 5
    assert row["independent_audit"]["exact"] is False
    assert row["independent_audit"]["sha256_forward"] != row["independent_audit"]["sha256_reverse"]
    assert row["live_bilateral_equation"][-1]["left_residual"]
    assert row["anti_shortcut"]["word_order_mirror"] is False


def test_novelty_preflight_rejects_duplicate_sweeps_and_artifact_matches():
    data = MODULE.run()
    saved = json.loads((ROOT / "runs/seed-benchmark-live-semantic-slot-expansion-20260916.json").read_text())
    assert data["novelty_preflight"]["status"] == "passed"
    assert data["novelty_preflight"]["duplicate_sweep_rejected"] is True
    assert saved["provenance"]["generator_sha256"] == data["provenance"]["generator_sha256"]
    assert data["next_repair"]["operator"]
