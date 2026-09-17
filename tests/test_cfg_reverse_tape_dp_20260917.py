from experiments.cfg_reverse_tape_dp_20260917 import audit, run

def test_audit_independent_hash_and_pointer():
    a = audit("A man, a plan, a canal: Panama")
    assert a["two_pointer_exact"] and a["sha_equal_under_reversal"]

def test_cfg_dp_reaches_long_frontier_and_records_chart():
    r = run()
    assert r["max_letters"] >= 100
    assert r["longest_frontier"]["audit"]["letters"] >= 100
    assert r["chart_states"] > 0
    assert r["novelty_preflight"]["near_miss_scoring"] is False
