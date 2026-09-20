from experiments.aspect_auxiliary_agreement_recipient_chart_20260920 import audit,chart,run
def test_aspect_and_agreement_states_are_conditioned():
 c=chart(); assert c['VP'] and c['DITRANS']; assert {'progressive','perfect'} <= {x.aspect for x in c['VP']}; assert {'sg','pl'} <= {x.number for x in c['NP']}
 assert all(x.valency=='ditransitive' for x in c['DITRANS'])
def test_complete_controls_have_independent_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['combines']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
