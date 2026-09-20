from experiments.semordnilap_grammar_intersection_20260929 import audit,run
def test_controls_are_real_words_and_audited():
 ex,co=run(2); rows=ex or co; assert rows and all(all(w.isalpha() for w in x['left_words']+x['right_words']) for x in rows)
def test_hash_pointer():
 a=audit('the river');assert a['pointer_mismatches']>0 and a['sha256_forward']!=a['sha256_reverse']
