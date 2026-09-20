from experiments.typed_question_answer_complement_scheduler_20260920 import audit,frames,run
def test_question_answer_frames_are_typed():
 fs=frames(); assert len(fs)>100; assert {'question','answer'} <= {x.kind for x in fs}; assert all(len(x.parts)==6 for x in fs)
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['baseline_is_control_only']; assert x['stats']['states']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
