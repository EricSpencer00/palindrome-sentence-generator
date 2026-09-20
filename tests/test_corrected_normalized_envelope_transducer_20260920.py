from experiments.corrected_normalized_envelope_transducer_20260920 import audit,run
def test_normalized_envelope():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['nodes']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
