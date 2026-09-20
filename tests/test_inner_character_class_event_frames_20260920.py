from inner_character_class_event_frames_20260920 import run,inner_state
def test_inner_gate_yields_complete_prose():
 x=run(); assert x['stats']['rendered_candidates']>0; assert all(r['complete_prose'] for r in x['rendered_candidates'])
def test_gate_is_distinct_and_pre_rendered():
 x=run(); assert x['stats']['states_rejected']>0; assert all(r['provenance']['inner_classes_enforced_before_render'] for r in x['rendered_candidates'])
def test_no_shortcuts():
 assert all(not r['provenance']['finished_tape_reversal'] and not r['provenance']['post_hoc_repair'] for r in run()['rendered_candidates'])
