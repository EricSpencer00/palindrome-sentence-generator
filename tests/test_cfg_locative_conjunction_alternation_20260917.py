import json
from pathlib import Path
def test_conjunction_lane():
 x=json.loads(Path('runs/cfg-locative-conjunction-alternation-20260917.json').read_text());assert x['control_count']==512 and x['repair_count']==512 and x['exact_count']==0
 for r in x['candidates']:
  assert r['locative_conjunction'] in ('and','as well as');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_order_repair():
 assert 'place roles' in json.loads(Path('runs/cfg-locative-conjunction-alternation-20260917.json').read_text())['next_repair']
