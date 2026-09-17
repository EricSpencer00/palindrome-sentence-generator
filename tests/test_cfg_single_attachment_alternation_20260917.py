import json
from pathlib import Path
def test_single_alternation_artifact():
 x=json.loads(Path('runs/cfg-single-attachment-alternation-20260917.json').read_text());assert x['control_count']==256 and x['alternation_count']==768 and x['exact_count']==0
 assert x['candidate_count']==1024
 for r in x['candidates']:
  assert r['novelty_preflight']['single_alternation_only'];assert r['anti_shortcut']['single_attachment_alternation'];assert r['audit']['two_pointer']==r['audit']['reverse_sha256_equal']
def test_next_repair_is_state_local():
 assert 'within each attachment state' in json.loads(Path('runs/cfg-single-attachment-alternation-20260917.json').read_text())['next_repair']
