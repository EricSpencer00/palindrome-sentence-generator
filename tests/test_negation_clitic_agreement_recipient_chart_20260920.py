from experiments.negation_clitic_agreement_recipient_chart_20260920 import audit,chart,run
def test_polarity_is_a_typed_auxiliary_state():
 c=chart(); assert c['VP'] and c['DITRANS']; assert {'positive','negative'} <= {x.polarity for x in c['VP']}; assert any(' not ' in f' {x.text} ' for x in c['VP'])
 assert all(x.valency=='ditransitive' for x in c['DITRANS'])
def test_controls_and_independent_audit():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['combines']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
