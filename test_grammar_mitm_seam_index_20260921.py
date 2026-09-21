from experiments.grammar_mitm_seam_index_20260921 import run
def test_mitm_records_equal_state_joins_and_audit():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['exact_gt38']==len(r['exact_candidates']); assert all('audit' in x for x in r['controls'])
