import json
from pathlib import Path
from experiments.reversible_morpheme_transducer_20260917 import run, EXPERIMENT_ID

def test_live_morpheme_transducer_audits_and_repair():
    d = run(); assert d["experiment_id"] == EXPERIMENT_ID
    assert d["status"] == "completed_diagnostic_unsound_for_palindrome_admission"
    assert d["novelty_preflight"]["fixed_tape_used"] is False
    assert all(x["provenance"]["authored_scene"] for x in d["candidates"])
    assert all(x["repair"]["heldout"] for x in d["candidates"])
    assert all("sha256_forward" in x["repair"]["audit"] for x in d["candidates"])
    states = [p for x in d["candidates"] for p in x["base"]["transducer"]["pairs"]]
    assert any(p["consumed_from_residual"] for p in states)
    assert all("residual_after" in p for p in states)
    assert d["soundness_audit"]["bilateral_outer_frontier_matching"] is False
    assert d["soundness_audit"]["admission_safe"] is False
    assert Path("runs/reversible-morpheme-transducer-20260917.json").exists()
