from experiments.dialogue_relative_clause_followup_20260916 import run
def test_dialogue_followup_changes_only_predicate():
 p=run();assert p['stats']['targeted_states']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert 'small key' in r['rendered'] and 'Jon said' in r['rendered'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
