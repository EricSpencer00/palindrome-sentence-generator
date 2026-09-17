import json
from pathlib import Path
def test_joint_place_lexical_lane():
 x=json.loads(Path('runs/cfg-joint-place-pair-lexical-20260917.json').read_text());assert x['control_count']==192 and x['repair_count']==64 and x['exact_count']==0
 for r in x['candidates']:
  assert '/' in r['place_pair'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_preposition_lexical():
 assert 'preposition lexical alternation' in json.loads(Path('runs/cfg-joint-place-pair-lexical-20260917.json').read_text())['next_repair']
