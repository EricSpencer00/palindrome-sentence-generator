import json
from pathlib import Path
def test_relative_object_lane():
 x=json.loads(Path('runs/cfg-attachment-relative-object-substitution-20260917.json').read_text());assert x['control_count']==3072 and x['repair_count']==1024 and x['exact_count']==0
 for r in x['candidates']:
  assert r['novelty_preflight']['state_specific_valency'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_determiner_repair():
 assert 'determiner alternation' in json.loads(Path('runs/cfg-attachment-relative-object-substitution-20260917.json').read_text())['next_repair']
