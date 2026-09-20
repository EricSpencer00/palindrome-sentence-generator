from experiments.shakespeare_complement_pronoun_reverse_lattice_20260920 import audit,grammar,run
def test_complement_and_pronoun_lattice():
 p,paths,s=grammar(); assert any('she' in x.text for x in p['COMP']); assert len(s)>100
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['states']>0
 for row in x['scene_controls']: assert audit(row['rendered'])==row['audit']
