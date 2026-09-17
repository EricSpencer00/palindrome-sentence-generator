import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/semantic-valency-trie-nfa-multichar-20260917.json'
def test_multichar_nfa_has_many_live_states():
 d=json.loads(RUN.read_text());assert d['expanded_states']>4 and len(d['live_state_sample'])>4
 assert d['expanded_states']<=d['budget'] and d['pruned_transitions']>=0
def test_no_complete_sentence_sweep_and_audit_contract():
 d=json.loads(RUN.read_text());assert 'live_state_sample' in d and 'terminal_paths' in d
 for p in d['terminal_paths']:
  assert isinstance(p['left'],str) and isinstance(p['right'],str)
