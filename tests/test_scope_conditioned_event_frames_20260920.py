from scope_conditioned_event_frames_20260920 import run,audit,realize,FRAMES
def test_typed_frames_render_complete_prose_and_audit():
 x=run(); assert x['stats']['rendered_candidates']>0; assert x['stats']['max_letters']>38
 assert all(r['complete_prose'] and len(r['audit']['sha256_forward'])==64 for r in x['rendered_candidates'])
def test_agreement_is_checked_before_surface():
 assert all(realize(f).split()[1].endswith('s') == (f['number']=='sg') for f in FRAMES)
def test_no_shortcuts():
 assert all(not r['provenance']['finished_tape_reversal'] and not r['provenance']['post_hoc_repair'] for r in run()['rendered_candidates'])
