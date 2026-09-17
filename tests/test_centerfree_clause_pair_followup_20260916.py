from experiments.centerfree_clause_pair_followup_20260916 import run
def test_joint_seam_followup_preserves_complete_clauses():
 p=run();assert p['stats']['new_states']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert r['repair_operator']['complete_clauses_preserved'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
