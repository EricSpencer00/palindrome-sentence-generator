import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/fixed-point-opposing-domain-propagation-20260917.json'
def test_fixed_point_and_audits():
 d=json.loads(RUN.read_text());assert d['candidate_count']==2
 for r in d['candidates']:
  assert r['fixed_point_rounds']>0 and r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_prose():
 d=json.loads(RUN.read_text());assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
