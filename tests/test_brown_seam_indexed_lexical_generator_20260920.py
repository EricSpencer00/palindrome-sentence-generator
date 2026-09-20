from experiments.brown_seam_indexed_lexical_generator_20260920 import audit,run
def test_seam_index_generator():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['states']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
