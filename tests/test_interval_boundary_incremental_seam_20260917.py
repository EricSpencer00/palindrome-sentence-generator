import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/interval-boundary-incremental-seam-20260917.json'
def test_intervals_and_audits():
 d=json.loads(RUN.read_text());assert d['candidate_count']==len(d['candidates'])>0
 for r in d['candidates']:
  assert r['intervals'] and r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_prose_and_layers():
 d=json.loads(RUN.read_text());assert len(d['layers'])==len(d['candidates'][0]['slots'])
 assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
