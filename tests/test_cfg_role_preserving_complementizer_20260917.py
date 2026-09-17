import json
from pathlib import Path
def test_role_complementizer_lane():
 x=json.loads(Path('runs/cfg-role-preserving-complementizer-20260917.json').read_text());assert x['control_count']==256 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert r['novelty_preflight']['role_preserved'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_locative_role_repair():
 assert 'locative state' in json.loads(Path('runs/cfg-role-preserving-complementizer-20260917.json').read_text())['next_repair']
