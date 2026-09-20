from experiments.broad_lexical_boundary_wfsa_20260925 import audit,run
def test_exact_only():assert all(x['audit']['two_pointer_exact'] for x in run(40,2))
def test_independent_hash_pointer():
 a=audit('the river');assert a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
