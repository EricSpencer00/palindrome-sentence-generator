import json
from pathlib import Path
def test_state_local_lane():
 x=json.loads(Path('runs/cfg-state-local-relative-verb-alternation-20260917.json').read_text());assert x['control_count']==768 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert r['novelty_preflight']['state_local_only'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_repair():
 assert 'subject substitution' in json.loads(Path('runs/cfg-state-local-relative-verb-alternation-20260917.json').read_text())['next_repair']
