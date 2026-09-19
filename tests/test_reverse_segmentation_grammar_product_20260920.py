from experiments.reverse_segmentation_grammar_product_20260920 import run, clauses, audit

def test_preflight_and_inventory_are_explicit():
    r=run(250000)
    assert r["novelty_preflight"]["status"]=="passed"
    assert r["config"]["left_forward_trie"] and r["config"]["right_reverse_trie"]
    assert r["config"]["inventory"]==len(tuple(clauses()))
    assert r["stats"]["truncated"] is False

def test_independent_audit_and_rejections_are_present():
    r=run(250000)
    assert r["provenance"]["independent_audits"]
    assert r["stats"]["rejected_rows"] == len(r["rejected_rows"])
    assert r["stats"]["exact_rows"] == len(r["exact_candidates"])
    assert audit("level")["two_pointer_exact"]
    assert audit("ordinary")["two_pointer_exact"] is False

def test_bounded_run_reports_frontier_without_claiming_global_completeness():
    r=run(2)
    assert r["stats"]["truncated"] is False
    assert "no-closure hypothesis" in r["falsifier"]
