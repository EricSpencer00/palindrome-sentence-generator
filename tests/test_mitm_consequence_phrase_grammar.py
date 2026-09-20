from mitm_consequence_phrase_grammar_20260920 import run
def test_mitm_frontier():
 x=run(); assert x['stats']['left_halves']==3 and x['stats']['right_halves']==3
 assert x['stats']['exact_gt38']==0; assert x['novelty_preflight']['mirror_pair_import'] is False
