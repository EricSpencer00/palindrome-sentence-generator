import json
from pathlib import Path

RUN = Path(__file__).parents[1] / 'runs/semantic-valency-trie-nfa-functionwords-20260917.json'

def test_function_word_states_expand_live_prefixes():
    d = json.loads(RUN.read_text())
    assert d['expanded_states'] > 8
    assert d['pruned_transitions'] >= 1
    assert 'bridge' in d['slots'] and 'determiner' in d['slots']

def test_diagnostics_have_independent_audit_and_no_shortcuts():
    d = json.loads(RUN.read_text())
    for c in d['rendered_candidates']:
        assert c['audit']['independent_two_pointer'] is False
        assert c['audit']['sha256'] and c['anti_shortcut']['intact_prose']
        assert not c['anti_shortcut']['mirrored_halves']
