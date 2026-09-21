import json
from pathlib import Path
from experiments.centerout_weighted_grammar_20260920 import audit, novelty_preflight, run

def test_preflight_and_frozen_run():
    assert novelty_preflight()["status"] == "passed"
    result=run(max_states=300)
    assert result["config"]["post_search_scoring"] is False
    assert result["provenance"]["finished_tape_reversal"] is False
    assert result["stats"]["obligations_checked"] > 0
    assert Path("runs/centerout-weighted-grammar-20260920.json").exists()

def test_audit_is_independent_and_exact_control_detected():
    row=audit("The writer marks a letter.")
    assert not row["pointer_exact"] and row["sha256_forward"] != row["sha256_reverse"]
