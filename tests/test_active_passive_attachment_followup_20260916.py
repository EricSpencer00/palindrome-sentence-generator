from experiments.active_passive_attachment_followup_20260916 import run
def test_attachment_followup_preserves_agent_patient_and_voice():
 p=run();assert p['stats']['new_states']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert 'engineer' in r['rendered'] and 'bridge' in r['rendered'] and 'beside the river' in r['rendered'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
