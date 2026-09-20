from experiments.recipient_case_alternant_scheduler_20260920 import audit,paths,run
def test_to_for_case_frames_and_unequal_paths():
 ps=paths(); assert {'to','for','none'} <= {x.case for x in ps}; assert len({len(x.words) for x in ps})>=3
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['seeded']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
