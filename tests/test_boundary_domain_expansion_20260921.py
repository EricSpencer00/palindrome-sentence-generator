from experiments.boundary_domain_expansion_20260921 import run

def test_expansion_carries_depth_and_renders_controls():
 r=run(); assert r['stats']['support_depth_frontier']>=2; assert r['rendered_candidates']; assert r['support_depth_frontier']['widening'].startswith('stopped')
def test_provenance_blocks_shortcuts():
 r=run(); assert r['novelty_preflight']['status']=='passed'; assert all(x['provenance']['finished_tape_reversal'] is False for x in r['rendered_candidates']); assert all('first_unsupported' in x for x in r['rendered_candidates'])
