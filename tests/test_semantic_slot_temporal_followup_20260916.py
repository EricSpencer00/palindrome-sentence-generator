from experiments.semantic_slot_temporal_followup_20260916 import run
def test_temporal_followup_preserves_gardener_slots():
 p=run();assert p['stats']['targeted_states']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert 'Before dusk' in r['rendered'] and 'near the glasshouse' in r['rendered'] and 'into the dry storehouse' in r['rendered'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
