import json,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; M=runpy.run_path(str(ROOT/'experiments/relative_locative_attachment_20260921.py'))
def test_locative_heldout_family():
 M['main'](); d=json.loads((ROOT/'runs/relative-locative-attachment-20260921.json').read_text())
 assert d['candidate_count']==4 and d['exact_count']==0 and d['reader_eligible'] is False
 assert d['stats']['longest_letters']>38
 for r in d['rendered_candidates']:
  assert r['shared_valency'] and r['audit']['forward_sha256']!=r['audit']['reverse_sha256']
def test_independent_mismatch(): assert M['audit']('A quiet school.')['two_pointer_exact'] is False
