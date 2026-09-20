from experiments.recipient_agreement_unequal_scheduler_20260920 import audit,paths,run
def test_number_conditioned_paths():
 ps=paths(); assert len({len(x.words) for x in ps})>=3; assert {'sg','pl'} <= {x.number for x in ps}; assert any('recipient' in x.roles for x in ps)
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['seeded']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
