from experiments.event_indexed_compositional_hypergraph_20260920 import audit,run
def test_event_hypergraph():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['graph_pair_states']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
