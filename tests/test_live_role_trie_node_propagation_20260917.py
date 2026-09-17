import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/live-role-trie-node-propagation-20260917.json'
def test_live_nodes_and_audits():
 d=json.loads(RUN.read_text());assert d['candidate_count']>0 and d['pruned_branches']>0
 for r in d['candidates']:
  assert r['live_trie_nodes'] and r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_no_shortcuts():
 d=json.loads(RUN.read_text());assert all(r['anti_shortcut']['intact_prose'] and not r['anti_shortcut']['mirrored_halves'] and r['novelty_preflight']['signature'] for r in d['candidates'])
