from experiments.semordnilap_poetic_clause_20261001 import audit,run
def test_exact_long_candidates_are_gated():
 assert all(x['audit']['two_pointer_exact'] and x['audit']['letters']>=38 for x in run(3))
def test_independent_audit():
 a=audit('the river');assert a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
