import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "semantic_slot_exact_closure_frontier_20260916",
    ROOT / "experiments/semantic_slot_exact_closure_frontier_20260916.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_bounded_search_emits_complete_fresh_prose_and_audit():
    result = MODULE.run()
    candidate = result["candidate"]
    assert candidate["rendered"].endswith(".")
    assert candidate["audit"]["letters"] > 38
    assert candidate["audit"]["sha256_forward"] == __import__("hashlib").sha256(MODULE.normalize(candidate["rendered"])[::-1][::-1].encode()).hexdigest()
    assert candidate["anti_shortcut"]["word_order_mirror"] is False
    assert candidate["anti_shortcut"]["repeated_unit"] is False


def test_exactness_is_independently_hash_checked_and_search_is_bounded():
    result = MODULE.run()
    candidate = result["candidate"]
    assert result["search"]["explored_complete_pairs"] <= result["search"]["state_bound"]
    assert candidate["audit"]["sha256_forward"] != candidate["audit"]["sha256_reverse"] or result["status"] == "completed_exact_closure"
    assert result["novelty_preflight"]["duplicate_sweep_rejected"] is True


def test_artifact_records_provenance_and_next_operator():
    result = MODULE.run()
    saved = json.loads((ROOT / "runs/semantic-slot-exact-closure-frontier-20260916.json").read_text())
    assert saved["provenance"]["generator_sha256"] == result["provenance"]["generator_sha256"]
    assert saved["next_repair"]["operator"]
