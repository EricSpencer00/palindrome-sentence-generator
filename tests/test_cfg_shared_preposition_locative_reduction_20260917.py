import json
from pathlib import Path
def test_shared_preposition_lane():
 x=json.loads(Path('runs/cfg-shared-preposition-locative-reduction-20260917.json').read_text());assert x['control_count']==512 and x['repair_count']==172 and x['exact_count']==0
 for r in x['candidates']:
  assert r['preposition_mode'] in ('split','shared');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_complementizer_reduction():
 assert 'complementizer reduction' in json.loads(Path('runs/cfg-shared-preposition-locative-reduction-20260917.json').read_text())['next_repair']
