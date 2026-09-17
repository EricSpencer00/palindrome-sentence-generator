import json
from pathlib import Path
def test_attachment_feature_artifact():
 x=json.loads(Path('runs/cfg-relative-attachment-feature-20260917.json').read_text());assert x['control_count'] and x['repair_count'] and x['exact_count']==0
 assert x['control_count'] == 3072 and x['repair_count'] == 1024
 for r in x['candidates']:
  assert r['anti_shortcut']['attachment_gated'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_repair_is_feature_carrying():
 assert 'attachment alternation' in json.loads(Path('runs/cfg-relative-attachment-feature-20260917.json').read_text())['next_repair']
