import json
from pathlib import Path
def test_verb_trie_agreement_lane():
 x=json.loads(Path('runs/cfg-midpoint-verb-trie-agreement-20260917.json').read_text()); assert x['candidate_count']==6144 and x['exact_count']==0
 for r in x['diagnostic_controls']:
  assert r['novelty_preflight']['verb_trie_intersection']; assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']; assert r['grammar_state']['midpoint_obligation']['all_lexical_tries_closed']
def test_no_shortcuts():
 x=json.loads(Path('runs/cfg-midpoint-verb-trie-agreement-20260917.json').read_text()); assert x['admitted_renderings']==[]
