import hashlib
import json
from pathlib import Path

from experiments.exact_by_construction_clause_attachment_20260917 import OUT, run


def test_clause_attachment_repair_is_a_real_lexical_edge_and_fail_closed():
    result = run(max_states=500_000)
    assert result["method"] == "exact_by_construction_optional_slot_clause_attachment"
    assert result["search"]["independent_clause_attachment"] is True
    assert result["search"]["rlaif_per_candidate"] is False
    assert result["status"] == "completed_no_admitted_closure"
    assert result["exact_candidates"] == []
    assert result["mechanically_admitted"] == []


def test_attachment_run_records_provenance_and_concrete_next_repair():
    result = json.loads(OUT.read_text())
    expected = hashlib.sha256(
        (Path(__file__).parents[1] / "experiments/exact_by_construction_clause_attachment_20260917.py").read_bytes()
    ).hexdigest()
    assert result["provenance"]["generator_sha256"] == expected
    assert result["provenance"]["seed_used_in_output"] is False
    assert result["next_repair"]["operator"] == "typed_relative_clause_attachment"
