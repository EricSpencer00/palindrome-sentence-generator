from experiments.variable_span_constraint_graph_20260921 import run, solve, independent_audit

def test_tiny_exhaustive_differential_and_audits():
    x=solve(limit=300)
    assert x['state_model']['variable_word_boundaries']
    assert x['state_model']['shared_character_variables']
    assert x['stats']['nodes'] <= 300 and x['stats']['learned_nogoods'] >= 0
    # differential audit is independent of the CSP's shared-character pruning
    for words in x['found']:
        assert independent_audit(words)['exact'] == (''.join(words)==''.join(words)[::-1])

def test_contract():
    x=run(limit=300)
    assert x['config']['lexical_entries']==48
    assert x['calibration_seed']['used_as_success'] is False
    assert x['novelty_preflight']['status']=='passed'
