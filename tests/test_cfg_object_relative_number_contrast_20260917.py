import json
from pathlib import Path
def test_number_contrast_artifact():
 x=json.loads(Path('runs/cfg-object-relative-number-contrast-20260917.json').read_text());assert x['control_count'] and x['repair_count'] and x['exact_count']==0
 for r in x['candidates']:
  q=r['agreement_state'];assert q['object_number']==q['object_determiner_number'];assert ' that ' in r['rendered'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_repair_is_determiner_gated():
 assert 'determiner alternation' in json.loads(Path('runs/cfg-object-relative-number-contrast-20260917.json').read_text())['next_repair']
