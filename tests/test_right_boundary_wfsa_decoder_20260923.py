from experiments.right_boundary_wfsa_decoder_20260923 import audit,run
def test_decoder_only_emits_exact_segmented_rows():
 for r in run(40,3): assert r['audit']['two_pointer_exact'] and r['right_words']
def test_hash_and_pointer_audit():
 a=audit('the river');assert a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
