from dependency_attachment_reset_20260920 import audit,run
def test_dependency_reset():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert r['stats']['rendered_candidates']>0; assert r['stats']['fresh_exact_gt38']==0
 assert all(x['provenance']['attachment_diagnostic_after_selection'] for x in r['rendered_candidates'])
def test_audit(): assert audit('A man, a plan, a canal: Panama!')['exact']
