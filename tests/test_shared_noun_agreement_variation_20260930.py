import hashlib
import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MOD = runpy.run_path(str(ROOT / "experiments/shared_noun_agreement_variation_20260930.py"))


def test_audit_independent_two_pointer_and_sha():
    text = "An aide rips nine memos; some men inspire Diana."
    au = MOD["audit"](text)
    tape = MOD["letters"](text)
    assert au["exact"] and MOD["pointer_exact"](text)
    assert au["sha256_forward"] == hashlib.sha256(tape.encode()).hexdigest()
    assert au["sha256_reverse"] == hashlib.sha256(tape[::-1].encode()).hexdigest()


def test_run_records_new_grammar_dimensions_and_controls():
    result = MOD["run"]()
    assert result["novelty_preflight"]["passed"]
    assert "determiner" in result["novelty_preflight"]["new_dimension"]
    assert result["controls"] and result["independent_validation"]
    assert result["exact_candidate_count"] == len(result["exact_candidates"])
    assert result["reader_eligible"] is False


def test_artifact_has_independent_audit_fields(tmp_path):
    result = MOD["run"](state_limit=1000)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    loaded = json.loads(path.read_text())
    assert loaded["method"].startswith("online shared-patient")
    assert loaded["novelty_preflight"]["posthoc_repair"] is False
