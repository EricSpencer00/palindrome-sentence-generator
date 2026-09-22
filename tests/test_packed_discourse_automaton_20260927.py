from experiments.packed_discourse_automaton_20260927 import event_slots, run
from experiments.packed_seam_grammar_20260927 import Grammar, intersect


def test_reference_requires_both_entities():
    assert event_slots(2, 0) is None
    assert event_slots(2, 1) is None
    assert event_slots(2, 2) is None
    assert event_slots(2, 3) is not None


def test_paragraph_recovery_and_complete_controls():
    result = run()
    assert result['positive_control']['recovered_in_both_conditions']
    assert all(not c['cap_reached'] for c in result['conditions'])
    assert len(result['controls']) == 4
    assert all(len(c['events']) == c['rendered'].count('.') for c in result['controls'])


def test_clause_boundary_does_not_require_local_palindrome():
    # Deliberately non-prose automaton fixture: neither individual clause is
    # palindromic, while their combined tape is exact across the boundary.
    g = Grammar()
    g.slot(('ab.',), 'clause1')
    g.slot(('ba.',), 'clause2')
    rows = intersect(g)['candidates']
    assert len(rows) == 1
    assert rows[0]['audit']['normalized'] == 'abba'
