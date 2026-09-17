import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/optional-relative-boundary-offset-20260917.json'
def test_relative_variants_and_audits():
 d=json.loads(RUN.read_text());assert d['candidate_count']==3
 assert {r['transition']['relative_clause'] for r in d['candidates']}=={'inserted','deleted'}
 for r in d['candidates']:
  assert r['offsets'] and r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_prose_and_novelty():
 d=json.loads(RUN.read_text());assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
