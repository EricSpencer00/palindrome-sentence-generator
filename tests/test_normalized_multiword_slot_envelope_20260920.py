from experiments.normalized_multiword_slot_envelope_20260920 import audit,run
def test_multiword_slots():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['nodes']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
