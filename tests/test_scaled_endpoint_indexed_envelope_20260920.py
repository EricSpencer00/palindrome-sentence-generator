from experiments.scaled_endpoint_indexed_envelope_20260920 import audit,run
def test_scaled_endpoint_envelope():
 x=run(); assert x['novelty_preflight']['status']=='passed'; assert x['stats']['nodes']>0; assert x['domain_sizes']['AGENT']>40
 for row in x['controls']: assert audit(row['rendered'])==row['audit']
