from experiments.broad_lexicon_envelope_transducer_20260920 import audit,run
def test_envelope_transducer():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['nodes']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
