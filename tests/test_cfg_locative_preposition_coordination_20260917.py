import json
from pathlib import Path
def test_preposition_coordination_lane():
 x=json.loads(Path('runs/cfg-locative-preposition-coordination-20260917.json').read_text());assert x['control_count']==589824 and x['repair_count']==147456 and x['exact_count']==0
 for r in x['candidates']:
  assert '/' in r['coordinated_prepositions'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_conjunction_repair():
 assert 'conjunction alternation' in json.loads(Path('runs/cfg-locative-preposition-coordination-20260917.json').read_text())['next_repair']
