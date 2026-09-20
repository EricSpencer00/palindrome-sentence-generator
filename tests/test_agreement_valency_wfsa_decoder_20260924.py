from experiments.agreement_valency_wfsa_decoder_20260924 import audit,valid_clause,run
def test_agreement_valency_frames():
 assert valid_clause(['the','pilot','sees','map']) and not valid_clause(['the','pilots','sees'])
def test_exact_outputs_only():
 assert all(x['audit']['two_pointer_exact'] for x in run(40,3))
def test_audit_independent():
 a=audit('the pilot sees map');assert a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
