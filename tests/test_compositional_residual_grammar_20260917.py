import json
from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_compositional_residual_artifact_scales_and_audits():
    data = json.loads((ROOT / "runs/compositional-residual-grammar-20260917.json").read_text())
    assert data["depths"] == [1, 2, 3, 4]
    assert [r["left_clauses"] for r in data["candidates"]] == [1, 2, 3, 4]
    assert data["candidates"][-1]["letters"] > data["candidates"][0]["letters"]
    assert all(r["independent_audit"]["letters"] == r["letters"] for r in data["candidates"])
    assert all(r["anti_shortcuts"]["finished_tape_reversal"] is False for r in data["candidates"])
    assert data["repair_after_failure"]["operator"]
