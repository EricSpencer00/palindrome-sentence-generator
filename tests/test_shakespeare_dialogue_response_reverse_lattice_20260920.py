from experiments.shakespeare_dialogue_response_reverse_lattice_20260920 import audit,grammar,run
def test_dialogue_response_lattice():
 p,paths,s=grammar(); assert {'asks','replies','answers','speaks'} <= {x.text.split()[0] for x in p['DIALOGUE']}; assert len(s)>100
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['states']>0
 for row in x['dialogue_controls']: assert audit(row['rendered'])==row['audit']
