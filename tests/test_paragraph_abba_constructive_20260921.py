from experiments.paragraph_abba_constructive_20260921 import run
def test_abba_renders_four_distinct_roles():
 r=run(); assert r['rendered_candidates']; assert all(x['gates']['distinct_sentence_roles'] for x in r['rendered_candidates'])
def test_obstruction_and_anti_shortcut_are_recorded():
 r=run(); assert r['obstruction']['status']=='blocked_at_first_character_cut'; assert r['novelty_preflight']['status']=='passed'; assert all(x['provenance']['finished_tape_reversal'] is False for x in r['rendered_candidates'])
