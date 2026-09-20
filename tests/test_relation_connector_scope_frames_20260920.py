from relation_connector_scope_frames_20260920 import run,audit,obligation
def test_relations_render_natural_complete_candidates():
 x=run(); assert x['stats']['relations']==3 and x['stats']['rendered_candidates']>0
 assert all(r['complete_prose'] and r['relation'] in ('contrast','cause','sequence') for r in x['rendered_candidates'])
def test_obligation_is_pre_render_diagnostic():
 assert obligation('the careful teacher marks the new route','a young pilot checks the clear signal') >= 0
 assert not audit('The careful teacher marks the new route, although a young pilot checks the clear signal.')['exact']
def test_shortcut_flags_closed():
 assert all(not r['provenance']['finished_tape_reversal'] and not r['provenance']['post_hoc_repair'] for r in run()['rendered_candidates'])
