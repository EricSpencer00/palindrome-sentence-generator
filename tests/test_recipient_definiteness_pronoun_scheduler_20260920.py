from experiments.recipient_definiteness_pronoun_scheduler_20260920 import audit,paths,run
def test_definite_indefinite_and_pronoun_states():
 ps=paths(); assert {'np','pron','none'} <= {x.recipient_kind for x in ps}; assert {'def','indef','pron','none'} <= {x.definiteness for x in ps}
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['seeded']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
