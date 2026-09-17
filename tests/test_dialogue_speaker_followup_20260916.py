from experiments.dialogue_speaker_followup_20260916 import run
def test_speaker_followup_preserves_dialogue_slots():
 p=run();assert p['stats']['targeted_states']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert 'Mara said' in r['rendered'] and 'small key' in r['rendered'] and 'stored' in r['rendered'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
