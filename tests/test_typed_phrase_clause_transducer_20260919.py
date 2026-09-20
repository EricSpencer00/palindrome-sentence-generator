from experiments.typed_phrase_clause_transducer_20260919 import consume, run
def test_residual_consumption():
 assert consume('abc','cba') == ('','')
 assert consume('abcd','cba') == ('d','')
 assert consume('abc','xyz') is None
def test_clause_lane():
 d=run(300)
 assert d['bank_sizes']['NP'] and d['bank_sizes']['VP'] and d['bank_sizes']['PP']
 assert d['candidate_count']==0
 assert d['status']=='rejected incomplete geometry'
