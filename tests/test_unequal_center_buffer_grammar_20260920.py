from unequal_center_buffer_grammar_20260920 import audit,run,step
def test_buffer_prunes_before_rendering():
 r=run(); assert r['stats']['frontier_pruned_before_rendering']>0; assert r['stats']['rendered_candidates']==0; assert r['stats']['fresh_exact_gt38']==0
def test_audit(): assert audit('A man, a plan, a canal: Panama!')['exact']
def test_reverse_facing_buffer_orientation():
    # Forward right prose ``ba`` is consumed from its reverse-facing end,
    # yielding ``ab`` to match the independently emitted left ``ab``.
    remainder, trace = step('ab', 'ba', '')
    assert remainder == '' and trace['mismatch'] is None
    remainder, trace = step('ac', 'ba', '')
    assert remainder is None and trace['mismatch'] == (1, 'c', 'b')
