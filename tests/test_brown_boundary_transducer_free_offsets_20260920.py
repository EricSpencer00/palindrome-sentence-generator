from experiments.brown_boundary_transducer_free_offsets_20260920 import audit,run
def test_free_offset_transducer_runs():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['nodes']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
