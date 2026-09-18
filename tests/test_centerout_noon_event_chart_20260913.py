from experiments.centerout_noon_event_chart_20260913 import replay, run

def test_grammatical_control_and_true_center():
    r=run(); assert r['config']['starts_at_center']; assert r['config']['equal_center_required']; assert r['seed_control']['independent_parse']; assert r['seed_control']['independent_exact_audit']['letters']>=30

def test_matched_ledger_replays_and_tampering_fails():
    r=run(); ledger=r['deepest_live_frontier']['ledger']; assert r['deepest_live_frontier']['independent_replay']['ok']; tampered=[dict(e) for e in ledger]; tampered[0]['char']='z'; assert replay(tampered)['ok'] is False

def test_all_exact_closures_are_replayed_and_admitted():
    r=run(); assert all(row['replay']['ok'] and row['mechanically_admitted'] for row in r['exact_closures']); assert r['exact_closures']==r['admitted_closures']
