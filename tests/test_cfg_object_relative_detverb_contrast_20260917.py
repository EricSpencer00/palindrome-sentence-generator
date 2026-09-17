import json
from pathlib import Path
def test_detverb_contrast_artifact():
 x=json.loads(Path('runs/cfg-object-relative-detverb-contrast-20260917.json').read_text());assert x['control_count'] and x['repair_count'] and x['exact_count']==0
 assert 'number-gated-determiner' in x['signature']
 for r in x['candidates']:
  assert r['anti_shortcut']['single_tree'] and ' that ' in r['rendered'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_attachment_operator():
 assert 'preposition state' in json.loads(Path('runs/cfg-object-relative-detverb-contrast-20260917.json').read_text())['next_repair']
