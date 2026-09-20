from experiments.paired_clause_lattice_20260927 import audit,run
def test_controls_are_independently_audited():
 ex,co=run(3);assert co and all('audit' in x for x in co)
def test_audit_hash_pointer():
 a=audit('the pilot maps the cove');assert a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
