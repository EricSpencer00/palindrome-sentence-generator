import json
from pathlib import Path

from llm_palindrome.validator import is_palindrome
from experiments.packed_consequence_reflection_boundary_20260930 import run


def test_boundary_lane_has_exact_gate_and_controls():
    result = run()
    assert result["provenance"]["complete_sentence_enumeration"] is False
    assert result["provenance"]["reverse_phrase_catalogue"] is False
    assert result["provenance"]["per_candidate_rlaif"] is False
    assert is_palindrome(result["positive_control"]["rendered"])
    assert all(not row["audit"]["exact"] for row in result["controls"])
    for row in result["candidates"]:
        assert row["audit"]["exact"]
        assert row["audit"]["independent_validator_exact"]


def test_artifact_is_current_and_reader_gate_is_closed():
    result = run()
    path = Path("runs/packed-consequence-reflection-boundary-20260930.json")
    assert path.exists()
    stored = json.loads(path.read_text()) if path.exists() else {}
    # The generator is authoritative; this lane never silently certifies
    # readability from lexical or programmatic scores.
    assert result["reader_test"]["status"] == "not_collected"
    assert stored.get("experiment_id") in (None, result["experiment_id"])
