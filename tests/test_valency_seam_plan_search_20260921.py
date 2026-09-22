from experiments.valency_seam_plan_search_20260921 import run
def test_valency_lane():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['plans']==16; assert r['stats']['exact_count']==0; assert r['stats']['longest_letters']>38
 for x in r['candidates']: assert x['provenance']['joint_valency_generation'] and x['provenance']['sentence_plan_gate']; assert x['grammar_plan_possible'] is True
def test_audits():
 for x in run()['candidates']: assert x['audit']['sha256_forward'] != x['audit']['sha256_reverse']; assert not x['audit']['two_pointer_exact']
