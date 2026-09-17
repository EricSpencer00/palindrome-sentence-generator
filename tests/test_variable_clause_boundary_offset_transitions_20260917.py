import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/variable-clause-boundary-offset-transitions-20260917.json'
def test_variable_boundaries_and_audits():
 d=json.loads(RUN.read_text());assert d['candidate_count']==len(d['candidates'])==4
 assert len({r['boundary_transition']['bridge'] for r in d['candidates']})==4
 for r in d['candidates']:
  assert r['direct_offsets'] and r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_and_prose():
 d=json.loads(RUN.read_text());assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
