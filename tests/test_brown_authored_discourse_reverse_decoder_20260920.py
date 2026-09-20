from experiments.brown_authored_discourse_reverse_decoder_20260920 import audit,run
def test_discourse_frames_audit():
 x=run(max_frames=300);assert x['novelty_preflight']['status']=='passed';assert x['stats']['complete_forward_frames']==300;assert x['stats']['reverse_states']>0
 for row in x['controls']:assert audit(row['rendered'])==row['audit']
