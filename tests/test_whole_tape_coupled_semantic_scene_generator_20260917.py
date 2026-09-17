import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/whole-tape-coupled-semantic-scene-generator-20260917.json'
def test_whole_tape_coupling_and_audits():
 d=json.loads(RUN.read_text());assert d['searched_pairs']==9 and d['candidate_count']>0
 for r in d['candidates']:
  assert r['left_bundle'] and r['right_bundle'] and r['audit']['exact']==r['audit']['independent_two_pointer'] and len(r['audit']['sha256'])==64
def test_provenance_no_shortcuts():
 d=json.loads(RUN.read_text());
 for r in d['candidates']:
  assert r['anti_shortcut']['intact_prose'] and not r['anti_shortcut']['mirrored_halves'] and r['novelty_preflight']['signature']
