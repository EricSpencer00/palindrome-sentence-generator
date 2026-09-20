from experiments.variable_boundary_lattice_decoder_20260926 import audit,lattice,run
def test_lattice_segments_variable_lengths():assert lattice('theman',{'the','man'},set())
def test_exact_audit_only():assert all(x['audit']['two_pointer_exact'] for x in run(40,2))
def test_pointer_sha():
 a=audit('the river');assert a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
