import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/edge-labeled-lexical-trie-semantic-20260917.json'
def test_trie_edges_and_audits():
 d=json.loads(RUN.read_text());assert d['candidate_count']==len(d['candidates'])==2 and d['rejected_prefix_transitions']>=0
 for r in d['candidates']:
  assert r['edge_labels'] and r['trie_options_before_completion']>=0
  assert r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_prose():
 d=json.loads(RUN.read_text());assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
