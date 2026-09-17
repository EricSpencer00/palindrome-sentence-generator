import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/joint-relative-opposing-seam-csp-20260917.json'
def test_csp_rows_and_independent_audits():
 d=json.loads(RUN.read_text());assert d['frontier_size']==81 and d['candidate_count']==12
 for r in d['candidates']:
  assert r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
  assert r['csp_score']>=0
def test_provenance_and_intact_prose():
 d=json.loads(RUN.read_text());assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
