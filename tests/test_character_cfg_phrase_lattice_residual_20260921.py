from experiments.character_cfg_phrase_lattice_residual_20260921 import audit, run

def test_cfg_lattice_emits_controls_and_live_pruning():
    result = run()
    assert result["novelty_preflight"]["status"] == "passed"
    assert result["stats"]["independent_pairs"] > result["stats"]["closed"]
    assert result["diagnostic_controls"]
    assert all("residual_trace" in row for row in result["reader_facing_candidates"])
    assert all(", while " in row["rendered"] and row["rendered"].endswith(".")
                for row in result["reader_facing_candidates"])

def test_pointer_and_hash_audit_agree():
    checked = audit("A man, a plan, a canal, panama.")
    assert checked["pointer_exact"]
    assert checked["sha256_forward"] == checked["sha256_reverse"]
