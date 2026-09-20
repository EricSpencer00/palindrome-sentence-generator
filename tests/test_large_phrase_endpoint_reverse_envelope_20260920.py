from experiments.large_phrase_endpoint_reverse_envelope_20260920 import audit,run
def test_large_phrase_envelope():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['states']>1000;assert x['stats']['controls']>=20
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
