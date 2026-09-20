from experiments.semordnilap_agreement_clause_20260930 import audit,run
def test_long_exact_outputs():assert all(x['audit']['two_pointer_exact'] and x['audit']['letters']>=38 for x in run(3))
def test_hash_pointer():
 a=audit('the river');assert a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
