from experiments.fresh_crossword_seam_csp_followup_20260916 import run
def test_followup_is_one_new_grammar_state():
 p=run();assert p['stats']['new_states']==1;assert p['novelty_preflight']['status']=='passed';assert p['novelty_preflight']['prior_assignments_replayed'] is False;r=p['candidate'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['prior_16_assignments_replayed'] is False
