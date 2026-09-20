from experiments.connector_clause_debt_lattice_20260928 import audit,run
def test_controls_and_debt_audit():
 ex,co=run(2);assert co and all('full_boundary_debt' in x for x in co)
def test_independent_hash_pointer():
 a=audit('pilot maps cove and poet is calm');assert a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
