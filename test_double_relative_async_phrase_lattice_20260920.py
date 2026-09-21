from double_relative_async_phrase_lattice_20260920 import run
def test_depth_two_controls():
 x=run(); assert x['stats']['rendered_controls']==4; assert x['stats']['online_states']>0
 assert all(r['provenance']['two_typed_relative_attachments'] and r['attachment_depth']==2 for r in x['controls'])
