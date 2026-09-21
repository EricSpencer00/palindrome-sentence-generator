import json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];M=runpy.run_path(str(R/'experiments/relative_plural_controller_20260921.py'))
def test_plural_controller():
 M['main']();d=json.loads((R/'runs/relative-plural-controller-20260921.json').read_text());assert d['candidate_count']==2 and d['exact_count']==0 and d['stats']['longest_letters']>38
 for x in d['rendered_candidates']:assert x['semantic_state']['plural_subject_controller']=='plural' and x['audit']['forward_sha256']!=x['audit']['reverse_sha256']
