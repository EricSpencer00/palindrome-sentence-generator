from experiments.heldout_endpoint_function_frames_20260920 import run,endpoint_gate,letters
def test_endpoint_gate_is_enforced_before_rendering():
 x=run(); assert x['stats']['rendered_candidates']>0
 assert all(endpoint_gate(r['left_frame'][0]+' '+r['left_frame'][1]+' '+r['left_frame'][2],r['right_frame'][0]+' '+r['right_frame'][1]+' '+r['right_frame'][2]) for r in x['rendered_candidates'])
def test_candidates_are_complete_and_independently_audited():
 assert all(r['complete_prose'] and len(r['audit']['sha256_forward'])==64 for r in run()['rendered_candidates'])
def test_no_repair_or_reversal():
 assert all(not r['provenance']['finished_tape_reversal'] and not r['provenance']['post_hoc_repair'] for r in run()['rendered_candidates'])
