from experiments.complete_frame_length_extension_20260920 import audit,frames,run
def test_fresh_frames_are_longer_than_baseline():
 fs=frames(); assert fs and min(len(x.parts) for x in fs)>=4; assert max(len(x.parts) for x in fs)>=5
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['baseline_is_control_only']; assert x['stats']['seams']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
