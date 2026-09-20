from experiments.authored_phrase_pair_graph_scene_expanded_20260920 import audit,run
def test_expanded_phrase_graph():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['left_chunks']==20;assert x['stats']['right_chunks']==20;assert x['stats']['composition_states']>0;assert x['baseline_control']['audit']['exact']
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
