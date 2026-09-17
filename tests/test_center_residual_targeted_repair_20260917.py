import hashlib, json, importlib.util
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location("target",ROOT/"experiments/center_residual_targeted_repair_20260917.py")
M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

def test_targeted_repair_is_held_out_and_preserves_outer_prose():
    r=M.run(); assert r["method"]["product_resweep"] is False
    assert r["stats"]["tested_centers"]==8
    assert r["parent"]["first_open"]["offset"]==0
    assert all(x["provenance"]["outer_clauses_preserved"] for x in r["candidates"])
    assert r["novelty_preflight"]["passed"]

def test_targeted_repair_has_independent_exact_and_shortcut_audits():
    r=M.run(); row=r["best_candidate"]; s=M.tape(row["rendered"]); p=row["independent_two_pointer"]
    assert p["sha256_forward"]==hashlib.sha256(s.encode()).hexdigest()
    assert p["sha256_reverse"]==hashlib.sha256(s[::-1].encode()).hexdigest()
    assert row["anti_shortcut"]["fixed_tape"] is False
    saved=json.loads((ROOT/"runs/center-residual-targeted-repair-20260917.json").read_text())
    assert saved["provenance"]["generator_sha256"]==r["provenance"]["generator_sha256"]
