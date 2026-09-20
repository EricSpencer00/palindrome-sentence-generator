from experiments.typed_tense_agreement_recipient_chart_20260920 import audit,chart,run
def test_feature_conditioned_chart_has_tense_number_valency():
 c=chart(); assert c['V'] and c['RECIP'] and c['DITRANS']
 assert {'present','past'} <= {x.tense for x in c['V']}
 assert {'sg','pl'} <= {x.number for x in c['NP']}
 assert all(x.valency=='ditransitive' for x in c['DITRANS'])
def test_controls_and_audit():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['combines']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
