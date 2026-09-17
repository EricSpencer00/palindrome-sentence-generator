import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/opposing-offset-lexical-choice-20260917.json'
def test_choice_rejection_and_audits():
 d=json.loads(RUN.read_text());assert d['candidate_count']==8
 for r in d['candidates']:
  assert r['chosen'] and r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_prose():
 d=json.loads(RUN.read_text());assert all(r['anti_shortcut']['intact_prose'] and r['novelty_preflight']['signature'] for r in d['candidates'])
