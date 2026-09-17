import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/joint-lexical-opposing-propagation-20260917.json'
def test_joint_trace_and_audits():
 d=json.loads(RUN.read_text());assert d['candidate_count']>0
 for r in d['candidates']:
  assert len(r['propagation_trace'])==8 and r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_prose():
 d=json.loads(RUN.read_text());assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
