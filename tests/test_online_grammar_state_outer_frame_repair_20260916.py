from experiments.online_grammar_state_outer_frame_repair_20260916 import run, EXPERIMENT_ID

def test_authored_outer_frame_repair_is_bounded_and_audited():
    p=run()
    assert p['experiment_id']==EXPERIMENT_ID
    assert p['stats']=={'rendered_probes':2,'exact':0,'mechanically_admitted':0,'reader_eligible':0}
    assert all(r['ledger_replay']['replayed'] for r in p['rendered_candidates'])
    assert all(len(r['sha256'])==64 and not r['two_pointer']['exact'] for r in p['rendered_candidates'])
