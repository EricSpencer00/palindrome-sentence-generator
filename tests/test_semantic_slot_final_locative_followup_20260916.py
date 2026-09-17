from experiments.semantic_slot_final_locative_followup_20260916 import run
def test_final_locative_followup_preserves_gardener_event():
 p=run();assert p['stats']['targeted_states']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert 'into the dry storehouse' in r['rendered'] and 'near the glasshouse' in r['rendered'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
