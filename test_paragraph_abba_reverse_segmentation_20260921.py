from experiments.paragraph_abba_reverse_segmentation_20260921 import audit,run
def test_abba_lane_has_independent_prose_and_audit():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['parses']==len(r['exact_candidates'])==0
 a=audit('The harbor pilot checks the tide. A quiet bell marks noon.')
 assert a['letters']>20 and a['sha256_forward'] != a['sha256_reverse_obligation']
