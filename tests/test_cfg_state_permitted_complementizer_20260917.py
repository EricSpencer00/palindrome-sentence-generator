import json
from pathlib import Path
def test_complementizer_lane():
 x=json.loads(Path('runs/cfg-state-permitted-complementizer-20260917.json').read_text());assert x['control_count']==3328 and x['repair_count']==256 and x['exact_count']==0
 for r in x['candidates']:
  assert (r['attachment_state']=='direct-object' and r['complementizer']=='that') or (r['attachment_state']=='locative' and r['complementizer']=='where');assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_direct_object_complementizer():
 assert 'that versus which' in json.loads(Path('runs/cfg-state-permitted-complementizer-20260917.json').read_text())['next_repair']
