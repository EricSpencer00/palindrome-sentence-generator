from experiments.fresh_seam_connector_followup_20260916 import run
def test_connector_child_preserves_prior_slots():
 p=run();assert p['stats']['child_states']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert 'chart beside the old pier' in r['rendered'] and 'for departure' in r['rendered'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
