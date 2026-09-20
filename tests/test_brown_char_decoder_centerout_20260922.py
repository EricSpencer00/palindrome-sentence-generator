from experiments.brown_char_decoder_centerout_20260922 import audit,generate
def test_immediate_mirror_is_exact():
 rows=generate(40,2); assert rows and all(r['audit']['two_pointer_exact'] for r in rows)
def test_independent_audit():
 a=audit('the river'); assert a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
