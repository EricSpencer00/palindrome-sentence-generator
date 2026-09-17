import json
from pathlib import Path
def test_joint_preposition_lexical_lane():
 x=json.loads(Path('runs/cfg-joint-preposition-lexical-20260917.json').read_text());assert x['control_count']==160 and x['repair_count']==96 and x['exact_count']==0
 for r in x['candidates']:
  assert r['preposition'] in ('near','beside','under');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_predicate_pair_repair():
 assert 'predicate pair substitution' in json.loads(Path('runs/cfg-joint-preposition-lexical-20260917.json').read_text())['next_repair']
