import json
from pathlib import Path
from experiments.semantic_center_sat_20260916 import run, SIGNATURE

def test_center_sat_is_real_prose_and_independently_rejected_until_exact():
    x = run()
    assert len(x["candidates"]) == 9
    assert x["stats"]["exact"] == 0
    assert all(c["provenance"]["center_authored"] for c in x["candidates"])
    assert all(c["exact_audit"]["hash_forward"] != c["exact_audit"]["hash_reverse"] for c in x["candidates"])
    assert all(c["rendered"].count(".") >= 2 for c in x["candidates"])
    assert x["novelty_preflight"]["exact_signature_collision"] is False

def test_center_sat_artifact_matches_schema():
    p = Path("runs/semantic-center-sat-20260916.json")
    assert p.exists()
    x = json.loads(p.read_text())
    assert x["signature"] == SIGNATURE
    assert x["next_repair"]
