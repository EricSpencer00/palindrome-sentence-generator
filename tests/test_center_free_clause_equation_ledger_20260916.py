import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "center_free_clause_equation_ledger_20260916",
    ROOT / "experiments/center_free_clause_equation_ledger_20260916.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_center_free_lane_emits_intact_long_prose_and_live_ledger():
    result = MODULE.run()
    row = result["best_candidate"]
    assert result["search"]["complete_realizations"] == 64
    assert row["letters"] > 100
    assert row["provenance"]["all_clauses_intact"] is True
    assert row["selection"]["center_was_not_fixed"] if "center_was_not_fixed" in row["selection"] else row["provenance"]["center_was_not_fixed"]
    assert row["residual_ledger"]["center_free"] is True


def test_center_free_lane_has_independent_exact_audits_and_no_shortcuts():
    result = MODULE.run()
    row = result["best_candidate"]
    normalized = MODULE.tape(row["rendered"])
    pointer = row["independent_two_pointer"]
    assert pointer["sha256_forward"] == hashlib.sha256(normalized.encode()).hexdigest()
    assert pointer["sha256_reverse"] == hashlib.sha256(normalized[::-1].encode()).hexdigest()
    assert row["independent_sha_agreement"] is False
    assert row["anti_shortcut"]["fixed_tape"] is False
    assert row["anti_shortcut"]["finished_surface_reversed"] is False
    assert result["novelty_preflight"]["passed"] is True
    assert result["novelty_preflight"]["duplicate_sweep_rejected"] is True


def test_center_free_artifact_provenance_and_next_repair_are_saved():
    result = MODULE.run()
    saved = json.loads((ROOT / "runs/center-free-clause-equation-ledger-20260916.json").read_text())
    assert saved["full_rendered_prose"] == result["full_rendered_prose"]
    assert saved["provenance"]["generator_sha256"] == result["provenance"]["generator_sha256"]
    assert saved["next_repair"]["operator"]
