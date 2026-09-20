from experiments.reverse_phrase_transducer_20260919 import consume, run
def test_phrase_residual():
 assert consume('ab','ba') == ('','')
 assert consume('abcd','cba') == ('d','')
 assert consume('ab','cd') is None
def test_phrase_run():
 d=run(300)
 assert d['reverse_index_keys']>0 and d['candidate_count']==0
