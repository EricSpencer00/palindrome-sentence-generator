import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "minimal_residual_grammar_repair_20260916",
    ROOT / "experiments/minimal_residual_grammar_repair_20260916.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_single_repair_preserves_agreement_and_complete_svo_prose():
    data = MODULE.run()
    row = data["candidates"][0]
    assert len(data["candidates"]) == 1
    assert row["repair"]["old_verb_object"] == "marks the distant buoy"
    assert row["repair"]["new_verb_object"] == "steers the small boat"
    assert all(row["grammar_checks"].values())
    assert row["anti_shortcut"]["intact_prose"]
    assert row["anti_shortcut"]["posthoc_character_edit"] is False


def test_repair_emits_independent_hashes_and_first_residual_progress():
    data = MODULE.run()
    row = data["candidates"][0]
    audit = row["independent_audit"]
    assert audit["exact"] is False
    assert audit["sha256_forward"] != audit["sha256_reverse"]
    assert row["residual"]["before"]["matched_outer_pairs"] == 0
    assert row["residual"]["after"]["matched_outer_pairs"] >= 1
    assert row["residual"]["after"]["closed"] is False
    assert data["novelty_preflight"]["single_targeted_state"]
    assert data["novelty_preflight"]["resweep_rejected"]


def test_run_artifact_is_the_single_targeted_state():
    data = MODULE.run()
    saved = json.loads((ROOT / "runs/minimal-residual-grammar-repair-20260916.json").read_text())
    assert saved["experiment_id"] == data["experiment_id"]
    assert saved["stats"] == {"rendered": 1, "exact": 0}
    assert saved["provenance"]["generator_sha256"] == data["provenance"]["generator_sha256"]
