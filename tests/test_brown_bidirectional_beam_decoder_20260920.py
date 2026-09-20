from experiments.brown_bidirectional_beam_decoder_20260920 import audit,run
def test_live_beam_has_complete_states_and_controls():
 x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['nodes']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
