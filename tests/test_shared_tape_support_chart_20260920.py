from experiments.shared_tape_support_chart_20260920 import chart, propagate, search, normalize, LEXICON, BINARY, UNARY

def test_seed_recovery_is_separate_calibration():
    from shared_tape_support_chart_20260920 import calibration_grammar
    result = search(38, 1000, *calibration_grammar())
    assert any(normalize(row['rendered']) == 'anaideripsninememossomemeninspirediana'
               for row in result['exact_candidates'])

def test_variable_boundaries_and_odd_center():
    result = search(5, lexicon={'A': ('ab',), 'B': ('cba',)},
                    binary=(('S', 'A', 'B'),), unary=())
    assert [r['rendered'] for r in result['exact_candidates']] == ['ab cba.']

def test_unreachable_lexical_support_is_removed():
    domains, _, _ = propagate([set('az'), set('az')],
                             {'A': ('a',), 'UNREACHABLE': ('zz',)},
                             (('S', 'A', 'A'),), ())
    assert domains == [{'a'}, {'a'}]

def test_ordinary_recursive_clause_is_in_language():
    tape = normalize('Diana writes a poem while Leon studies a map')
    assert ('S', 0, len(tape)) in chart([{c} for c in tape])

def test_exhaustive_tiny_language_matches_solver():
    lex = {'A': ('ab', 'ba', 'aa'), 'B': ('aba', 'cba', 'aab')}
    expected = {a+' '+b+'.' for a in lex['A'] for b in lex['B']
                if a+b == (a+b)[::-1] and a != a[::-1] and b != b[::-1]
                and a != b}
    result = search(5, 1000, lex, (('S', 'A', 'B'),), ())
    assert not result['stats']['truncated']
    assert {x['rendered'] for x in result['exact_candidates']} == expected
