from experiments.packed_seam_grammar_20260927 import Grammar, SEED, intersect, norm, run


def test_packed_solver_recovers_incumbent():
    result = run()
    assert not result['cap_reached']
    assert any(row['audit']['normalized'] == norm(SEED) for row in result['candidates'])
    assert all(row['connected'] and row['audit']['exact'] for row in result['candidates'])


def test_odd_even_and_epsilon_paths():
    g = Grammar()
    g.slot(('a',), 'first')
    g.slot(('', 'b', 'bc'), 'middle')
    g.slot(('a',), 'last')
    rows = intersect(g)['candidates']
    assert {row['audit']['normalized'] for row in rows} == {'aa', 'aba'}


def test_word_boundaries_need_not_match():
    g = Grammar()
    g.slot(('a', 'ab'), 'first')
    g.slot(('ba', 'a', 'b'), 'second')
    assert {row['audit']['normalized'] for row in intersect(g)['candidates']} == {'aa', 'aba', 'abba'}
