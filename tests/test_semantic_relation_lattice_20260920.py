from semantic_relation_lattice_20260920 import audit,run
def test_relation_gate_pre_render():
 r=run(); assert r['stats']['compatible_states']>0; assert r['stats']['rejected_incompatible_before_render']>0; assert r['stats']['fresh_exact_gt38']==0
 assert all(x['provenance']['semantic_signature_gated_before_render'] for x in r['rendered_candidates'])
def test_audit(): assert audit('A man, a plan, a canal: Panama!')['exact']
