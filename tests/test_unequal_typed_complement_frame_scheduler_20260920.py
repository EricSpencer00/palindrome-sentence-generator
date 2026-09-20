from experiments.unequal_typed_complement_frame_scheduler_20260920 import audit,frames,run
def test_typed_complement_frames_are_complete_and_variable():
 fs=frames(); assert len(fs)>100; assert len({len(x.parts) for x in fs})>=2; assert {x.comp_type for x in fs}=={'finite_complement','finite_complement_adjunct'}
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['baseline_is_control_only']; assert x['stats']['states']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
