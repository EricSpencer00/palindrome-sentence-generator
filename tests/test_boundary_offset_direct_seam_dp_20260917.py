import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/boundary-offset-direct-seam-dp-20260917.json'
def test_direct_offsets_and_audit():
 d=json.loads(RUN.read_text());assert d['candidate_count']==len(d['candidates'])>0
 for r in d['candidates']:
  assert len(r['direct_offsets'])==len(r['slots'])
  assert r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_new_operator_metadata():
 d=json.loads(RUN.read_text());assert len(d['layers'])==len(d['candidates'][0]['slots'])
 assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
