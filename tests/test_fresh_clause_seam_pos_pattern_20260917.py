from experiments.fresh_clause_seam_pos_pattern_20260917 import run
def test_fresh_clause_seam():
 x=run();assert x['novelty_preflight']['status']=='passed';assert all(p['audit']['exact'] and p['audit']['letters']>38 for p in x['candidates'])
