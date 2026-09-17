from experiments.semantic_slot_adjunct_followup_20260916 import run
def test_adjunct_followup_preserves_other_slots():
 p=run();assert p['stats']['targeted_states']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert 'near the glasshouse' in r['rendered'] and 'wrapped bundle' in r['rendered'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
