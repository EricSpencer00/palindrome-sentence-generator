from experiments.dialogue_reverse_grammar_20260920 import run
def test_dialogue_gate():
 x=run(); assert x['novelty_preflight']['status'] in ('passed','zero-frontier')
 assert all(r['audit']['exact'] for r in x['rendered_candidates'])
