import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/character-labeled-prefix-trie-gate-20260917.json'
def test_character_requirements_and_audits():
 d=json.loads(RUN.read_text());assert d['candidate_count']==3
 for r in d['candidates']:
  assert isinstance(r['character_requirements'], list) and r['left_trie_prefixes'] and r['right_trie_prefixes'] and r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_prose():
 d=json.loads(RUN.read_text());assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
