from experiments.semantic_role_matrix_adjunct_temporal6_csp_20260920 import run

def test_dual_temporal_attachment_is_audited():
    x = run(); s = x['stats']
    assert s['heldout_clause_paths'] == 48
    assert s['paired_grammar_states'] == 2304
    assert s['rendered_controls'] == 24
    assert s['longest_rendered_control_letters'] >= 150
    assert s['mechanical_exact_candidates'] == 0
    assert s['exact_clean_above_38'] == 0
    assert x['novelty_preflight']['registry_inspected']
    assert all(r['provenance']['post_hoc_repair'] is False for r in x['rendered_controls'])
