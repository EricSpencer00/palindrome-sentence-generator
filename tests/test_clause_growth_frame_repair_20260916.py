import json
from pathlib import Path
from experiments.clause_growth_frame_repair_20260916 import SIGNATURE, run

def test_clause_growth_has_real_prose_and_independent_rejection():
    x = run()
    assert x["novelty_preflight"]["exact_signature_collision"] is False
    assert x["stats"] == {"candidates": 5, "exact": 0, "admitted": 0}
    assert [len(c["frames"]) for c in x["candidates"]] == [1, 2, 3, 2, 4]
    assert all(c["provenance"]["ordinary_order"] for c in x["candidates"])
    assert all(c["exact_audit"]["hash_forward"] != c["exact_audit"]["hash_reverse"] for c in x["candidates"])
    assert x["candidates"][-1]["provenance"]["repair_from"] == "growth-3-clause"
    assert x["candidates"][-1]["frontier_selection"]["tested_frames"] == 3

def test_clause_growth_artifact_matches_registry_contract():
    p = Path("runs/clause-growth-frame-repair-20260916.json")
    assert p.exists()
    assert json.loads(p.read_text())["signature"] == SIGNATURE
