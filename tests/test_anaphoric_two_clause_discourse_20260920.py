from experiments.anaphoric_two_clause_discourse_20260920 import audit,run
def test_anaphoric_discourse_controls_and_novelty():
    x=run();assert x['novelty_preflight']['status']=='passed';assert x['stats']['controls']>=20;assert x['stats']['bound_frames']>0
    for row in x['controls']:assert audit(row['rendered'])==row['audit']
    assert all('they reviews' not in row['rendered'] and 'they opens' not in row['rendered'] for row in x['controls'])
