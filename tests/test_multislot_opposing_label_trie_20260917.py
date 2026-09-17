import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/multislot-opposing-label-trie-20260917.json'
def test_multislot_prefix_propagation():
 d=json.loads(RUN.read_text());assert d['candidate_count']==2 and d['frontier_size']==16
 for r in d['candidates']:
  assert len(r['trie_prefix_states'])==6 and r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_and_prose():
 d=json.loads(RUN.read_text());assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
