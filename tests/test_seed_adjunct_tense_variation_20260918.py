from experiments.seed_adjunct_tense_variation_20260918 import independent_audit, run

def test_controls_have_independent_hash_pointer_audits_and_withheld_seed_is_not_output():
    payload = run()
    assert payload["construction"]["semantic_adjunct_state"]
    assert payload["construction"]["tense_state"]
    assert payload["novelty_preflight"]["prior_lane_reused"] is False
    for row in payload["rendered_candidates"]:
        assert row["audit"] == independent_audit(row["rendered"])
        assert row["provenance"]["withheld_seed_used_as_output"] is False

def test_search_has_live_equation_nodes_before_any_rendered_closure():
    payload = run()
    assert all("equation" in row for row in payload["search"]["nodes"])
    assert payload["reader_gate"]["programmatic_metrics_are_diagnostic"]
