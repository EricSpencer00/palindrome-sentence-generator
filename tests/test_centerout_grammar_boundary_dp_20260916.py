import json
from pathlib import Path

from experiments.centerout_grammar_boundary_dp_20260916 import run, EXPERIMENT_ID

ROOT = Path(__file__).resolve().parents[1]

def test_centerout_boundary_dp_has_novelty_and_independent_audits():
    payload = run()
    assert payload["novelty_preflight"]["passed"]
    assert payload["stats"] == {"candidates": 10, "exact": 0, "mechanically_admitted": 0}
    assert all("exact_audit" in row and "frontier" in row for row in payload["candidates"])
    assert all(row["provenance"]["finished_sentence_reversed"] is False for row in payload["candidates"])
    registry = json.loads((ROOT / "docs/experiment-novelty-registry.json").read_text())
    assert any(row["id"] == EXPERIMENT_ID for row in registry["entries"])

def test_boundary_dp_candidate_is_intact_prose_and_long_enough():
    row = run()["candidates"][-1]
    assert row["letters"] >= 39
    assert row["rendered"].endswith(".")
    assert row["exact_audit"]["two_pointer_exact"] is False
