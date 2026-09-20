from experiments.unequal_complete_frame_seam_scheduler_20260920 import audit,frames,run
def test_frames_have_variable_lengths():
 assert len({len(x.parts) for x in frames()})>=2
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['baseline_is_control_only']; assert x['stats']['states']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
