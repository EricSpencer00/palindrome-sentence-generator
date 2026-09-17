from experiments.fresh_crossword_seam_csp_20260916 import run
def test_fresh_csp_has_no_fixed_tape_and_audits_prose():
 p=run();assert p['stats']['assignments']==16;assert p['novelty_preflight']['status']=='passed';assert p['novelty_preflight']['fixed_tape_or_seed_guidance'] is False
 for r in p['candidates']:
  assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['fixed_tape_used'] is False
