import json
from pathlib import Path
def test_order_swap_lane():
 x=json.loads(Path('runs/cfg-locative-complement-order-swap-20260917.json').read_text());assert x['control_count']==293 and x['repair_count']==250 and x['exact_count']==0
 for r in x['candidates']:
  assert r['attachment_state']=='locative-in-which';assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_reduction_repair():
 assert 'complement reduction' in json.loads(Path('runs/cfg-locative-complement-order-swap-20260917.json').read_text())['next_repair']
