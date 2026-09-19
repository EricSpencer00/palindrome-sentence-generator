from experiments import large_lexicon_heldout_verbs_20260920 as lane
def test_heldout_bank_and_controls():
    out=lane.run(30)
    assert out['novelty_preflight']['status']=='novel'
    assert out['stats']['heldout_verbs']>=20 and out['stats']['states']==30
    assert out['controls'] and all(r['trie_intersection'] for r in out['controls'])
    assert all(r['orbit_assignment'] for r in out['controls'])
    assert all(r['audit']['sha256_forward']!=r['audit']['sha256_reverse'] for r in out['controls'])
def test_no_shortcuts():
    p=lane.run(2)['provenance']
    assert p['ordinary_order_complete_clauses'] and p['word_boundaries_before_render']
    assert not p['post_hoc_repair'] and not p['word_order_mirror'] and not p['repeated_modules']
