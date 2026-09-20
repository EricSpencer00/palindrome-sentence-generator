from experiments.phrase_boundary_indexed_centerout_20260920 import audit, index_pairs, run

def test_index_is_keyed_before_rendering():
    ix=index_pairs(); assert ix and all(len(k)==3 for k in ix)

def test_clean_no_closure_or_exact_audit():
    p=run(5,500); assert p['pair_options']>0 and p['closures']==len(p['candidates'])

def test_audit_has_independent_pointer_and_hashes():
    a=audit('a baker'); assert a['two_pointer_exact'] is False and a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
