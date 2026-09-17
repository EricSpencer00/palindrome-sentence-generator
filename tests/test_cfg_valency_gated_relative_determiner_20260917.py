import json
from pathlib import Path
def test_valency_determiner_lane():
 x=json.loads(Path('runs/cfg-valency-gated-relative-determiner-20260917.json').read_text());assert x['control_count']==3328 and x['repair_count']==3328 and x['exact_count']==0
 for r in x['candidates']:
  assert r['novelty_preflight']['valency_gated'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_complementizer_repair():
 assert 'complementizer alternation' in json.loads(Path('runs/cfg-valency-gated-relative-determiner-20260917.json').read_text())['next_repair']
