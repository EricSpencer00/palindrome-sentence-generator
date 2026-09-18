from experiments.centerout_crossword_seam_chart_20260913 import CONTROL, replay, run

def test_control_is_grammatical_and_long_enough():
 r=run();assert r['seed_control']['rendered']==CONTROL;assert r['seed_control']['independent_parse'];assert r['seed_control']['independent_exact_audit']['letters']>=30
def test_every_live_ledger_replays_and_tampering_fails():
 r=run();l=r['deepest_live_frontier']['ledger'];assert r['deepest_live_frontier']['independent_replay']['ok'];
 if l:
  bad=[dict(e) for e in l];bad[0]['char']='z';assert not replay(bad)['ok']
def test_exact_closures_are_gate_checked():
 r=run();assert all(x['independent_exact_audit']['exact'] and x['mechanically_admitted'] for x in r['exact_closures']);assert r['admitted_closures']==r['exact_closures']
