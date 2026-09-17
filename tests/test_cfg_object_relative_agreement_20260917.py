import json
from pathlib import Path

def test_object_relative_agreement_artifact():
 x=json.loads(Path('runs/cfg-object-relative-agreement-20260917.json').read_text())
 assert x['control_count']>0 and x['repair_count']>0 and x['exact_count']==0
 for r in x['candidates']:
  assert ' that ' in r['rendered'] and r['anti_shortcut']['single_tree']
  assert r['agreement_state']['transition'].startswith('NP->VP requires V[')
  assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']

def test_next_operator_is_number_gated():
 x=json.loads(Path('runs/cfg-object-relative-agreement-20260917.json').read_text())
 assert 'number contrast' in x['next_repair']
