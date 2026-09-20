from experiments.typed_center_seam_complement_product_20260920 import audit,frames,run
def test_complete_complement_frames_precede_seam():
 fs=frames(); assert len(fs)>=100; assert all(len(x.parts)==3 for x in fs)
def test_controls_and_audits():
 x=run(state_limit=12000); assert x['novelty_preflight']['endpoint_seed'] is False; assert x['stats']['seams']>0
 for row in x['complete_prose_controls']: assert audit(row['rendered'])==row['audit']
