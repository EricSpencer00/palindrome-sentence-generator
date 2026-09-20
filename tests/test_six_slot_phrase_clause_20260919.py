from experiments.six_slot_phrase_clause_20260919 import run
def test_six_slot_geometry_is_explicit_and_fail_closed():
 d=run(200)
 assert tuple(d['roles'])==('NP','VP','PP','PP','VP','NP')
 assert d['candidate_count']==0
