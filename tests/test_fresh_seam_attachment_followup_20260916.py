from experiments.fresh_seam_attachment_followup_20260916 import run
def test_attachment_followup_preserves_prior_seam():
 p=run();assert p['stats']['heldout_repairs']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert 'chart beside the old pier' in r['rendered'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
