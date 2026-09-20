from experiments.authored_phrase_pair_graph_scene_20260920 import audit,run
def test_phrase_graph_and_baseline():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['composition_states']>0;assert x['baseline_control']['audit']['exact']
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
