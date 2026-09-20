from experiments.inflectional_clitic_boundary_csp_20260920 import audit, run

def test_seed_audit():
    assert audit('An aide rips nine memos; some men inspire Diana.')['exact']

def test_fail_closed_controls_and_audits():
    out=run()
    assert out['stats']['fresh_exact_gt38']==0
    assert all(x['audit']['sha_equal'] is False for x in out['controls'])
