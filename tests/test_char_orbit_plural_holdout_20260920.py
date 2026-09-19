from experiments.char_orbit_plural_holdout_20260920 import run

def test_plural_holdout_is_bounded_and_agreement_is_live():
    result=run()
    assert result['novelty_preflight']['status']=='passed'
    assert result['stats']['visited']==32
    assert all(r['complete_clause_gate']['agreement'] for r in result['complete_prose_controls'])
    assert all(r['semantic_states'][0]['number']=='plural' if r['words'][0]=='wardens' else r['semantic_states'][0]['number']=='singular' for r in result['candidates'])

def test_shortcuts_and_rlaif_are_banned():
    result=run()
    assert result['failure_and_next_discriminator']['rlaif']=='not used'
    assert all(not r['provenance']['finished_tape_reversed'] and not r['provenance']['word_order_symmetry'] for r in result['candidates'])
