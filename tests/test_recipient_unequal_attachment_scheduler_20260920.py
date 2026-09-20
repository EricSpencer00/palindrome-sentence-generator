from experiments.recipient_unequal_attachment_scheduler_20260920 import audit,paths,run
def test_recipient_paths_have_unequal_attachments():
 ps=paths(); assert len({len(x.words) for x in ps})>=3; assert any('recipient' in x.roles for x in ps); assert any('adjunct' in x.roles for x in ps)
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['seeded']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
