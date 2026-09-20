from cross_word_boundary_grammar_dp_20260920 import run,boundary_step
def test_cross_word_transitions_carry_residual():
 x=run(); assert x['stats']['transition_states']>0; assert all(r['cross_word_trace'] for r in x['rendered_candidates'])
def test_complete_prose_and_audit():
 assert all(r['complete_prose'] and len(r['audit']['sha256_forward'])==64 for r in run()['rendered_candidates'])
def test_no_tape_shortcut():
 assert all(not r['provenance']['finished_tape_reversal'] and not r['provenance']['post_hoc_repair'] for r in run()['rendered_candidates'])
