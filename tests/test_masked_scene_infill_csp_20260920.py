from experiments.masked_scene_infill_csp_20260920 import audit,run
def test_masked_scene_csp():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['length_band_states']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
