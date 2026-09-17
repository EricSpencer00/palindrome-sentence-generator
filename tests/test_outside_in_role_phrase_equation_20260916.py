import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "outside_in_role_phrase_equation_20260916",
    ROOT / "experiments/outside_in_role_phrase_equation_20260916.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_fresh_outside_in_bank_emits_long_complete_prose():
    result = MODULE.run()
    row = result["best_candidate"]
    assert len(result["full_rendered_prose"]) > 100
    assert row["audit"]["letters"] >= 100
    assert row["provenance"]["choices_before_rendering"] is True
    assert row["anti_shortcut"]["catalogue_imported"] is False


def test_pointer_and_sha_audits_are_independent_and_mechanical_gate_is_present():
    result = MODULE.run()
    row = result["best_candidate"]
    normalized = MODULE.tape(row["rendered"])
    assert row["audit"]["sha256_forward"] == hashlib.sha256(normalized.encode()).hexdigest()
    assert row["audit"]["sha256_reverse"] == hashlib.sha256(normalized[::-1].encode()).hexdigest()
    assert row["audit"]["exact"] is False or row["audit"]["sha256_forward"] == row["audit"]["sha256_reverse"]
    assert "exact_letter_palindrome" in row["mechanical_admission"]
    assert result["novelty_preflight"]["duplicate_sweep_rejected"] is True


def test_artifact_has_rendered_prose_provenance_and_next_operator():
    result = MODULE.run()
    saved = json.loads((ROOT / "runs/outside-in-role-phrase-equation-20260916.json").read_text())
    assert saved["full_rendered_prose"] == result["full_rendered_prose"]
    assert saved["provenance"]["generator_sha256"] == result["provenance"]["generator_sha256"]
    assert saved["next_repair"]["operator"]
