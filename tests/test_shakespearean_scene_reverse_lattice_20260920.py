from experiments.shakespearean_scene_reverse_lattice_20260920 import audit,grammar,run
def test_fresh_scene_vocabulary_and_complete_paths():
 p,paths,s=grammar(); assert all(any(x in q.text for q in p['NP']) for x in ('bard','king','queen')); assert len(s)>100
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['states']>0
 for row in x['rendered_candidates'][:20]: assert audit(row['rendered'])==row['audit']
