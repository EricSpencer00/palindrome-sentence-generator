from experiments.weighted_grammar_automaton_20260920 import audit, run

def test_audit_has_independent_pointers_and_sha():
    a = audit("A quiet poet")
    assert a["letters"] == 10 and len(a["sha256_forward"]) == 64
    assert a["sha256_forward"] != a["sha256_reverse"]

def test_run_records_real_partials_and_online_method():
    r = run(max_states=12000, beam=80)
    assert r["stats"]["memoized_residuals"] > 0
    assert r["near_misses"] or r["exact_candidates"]
    assert r["novelty_preflight"]["status"] == "passed"
