import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from experiments.luna_lm_obligation_clause_search_20260917 import OUT, run

ROOT = Path(__file__).parents[1]


def test_run_has_actual_candidates_and_no_exact_readability_claim():
    result = run()
    assert result["status"] == "completed_no_exact_closure"
    assert len(result["candidates"]) == 12
    assert result["reader_eligible"] is False
    assert result["novelty_preflight"]["status"] == "passed"
    assert all(row["provenance"]["choices_before_rendering"] for row in result["candidates"])


def test_pointer_and_sha_agree_and_shortcuts_are_rejected():
    result = json.loads(OUT.read_text())
    assert all(row["independent_audit_agreement"] for row in result["candidates"])
    assert all(not any(row["anti_shortcut"].values()) for row in result["candidates"])
    assert all(row["semantic_role_states"]["left"] == ["agent", "transitive_event", "theme", "adjunct_setting"] for row in result["candidates"])
    assert result["next_repair"]["operator"]


def test_artifact_records_independent_generator_pointer():
    result = json.loads(OUT.read_text())
    expected = hashlib.sha256((ROOT / "experiments/luna_lm_obligation_clause_search_20260917.py").read_bytes()).hexdigest()
    assert result["provenance"]["generator_sha256"] == expected
    assert all(row["independent_pointer_audit"]["algorithm"] == "independent_two_pointer" for row in result["candidates"])
