from experiments.centerout_museum_heldout_repair_20260916 import run
def test_single_museum_repair_preserves_center():
 p=run();assert p['stats']['new_states']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert 'audience listens while' in r['rendered'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
