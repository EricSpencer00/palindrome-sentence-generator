import json
from pathlib import Path
RUN=Path(__file__).parents[1]/'runs/function-word-trie-boundary-seam-20260917.json'
def test_function_word_rows_audited():
 d=json.loads(RUN.read_text()); assert d['candidate_count']==len(d['candidates'])>0
 for r in d['candidates']:
  assert r['character_checks']>0 and r['audit']['exact']==r['audit']['independent_two_pointer']
  assert len(r['audit']['sha256'])==64 and r['novelty_preflight']['signature']
def test_intact_prose_flags():
 d=json.loads(RUN.read_text()); assert all(r['anti_shortcut']['intact_prose'] for r in d['candidates'])
