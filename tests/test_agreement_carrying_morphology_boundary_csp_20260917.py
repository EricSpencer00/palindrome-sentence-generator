import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "agreement_carrying_morphology_boundary_csp_20260917",
    ROOT / "experiments/agreement_carrying_morphology_boundary_csp_20260917.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_preflight_is_novel_and_product_is_bounded():
    result = MODULE.novelty_preflight()
    assert result["status"] == "passed"
    assert result["bounded_product"]
    assert not result["signature_overlaps"]


def test_inflectional_transducer_rejects_singular_plural_mismatch():
    singular = MODULE.inflections("sg", "present")[0]
    plural = MODULE.inflections("pl", "present")[0]
    assert singular.finite.endswith("s")
    assert not plural.finite.endswith("s")


def test_candidates_are_complete_ordinary_clauses_and_not_shortcuts():
    row = MODULE.transduce(MODULE.complete_clauses()[0], MODULE.complete_clauses()[-1])
    assert row is not None
    assert row["rendered"].count(".") == 2
    assert all(value is False for value in row["anti_shortcut_flags"].values())
    assert row["audit"]["letters"] >= 50


def test_audit_uses_two_pointer_and_independent_forward_reverse_hashes():
    row = MODULE.audit("The baker carries a bell near the harbor.")
    assert row["independent_two_pointer_exact"] is False
    assert row["sha256_forward"] != row["sha256_reverse"]
    assert row["two_pointer_mismatches"]


def test_run_records_failure_and_concrete_held_out_repair():
    result = MODULE.run()
    assert result["candidate_count"] == result["stats"]["transduced"]
    assert result["failure_and_repair"]["operator"]
    assert result["failure_and_repair"]["held_out_variants"]
    out = ROOT / "runs" / (MODULE.EXPERIMENT_ID + ".json")
    out.write_text(json.dumps(result, indent=2) + "\n")
