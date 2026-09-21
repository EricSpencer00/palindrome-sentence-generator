import json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];M=runpy.run_path(str(R/'experiments/relative_instrument_morphology_seam_20260921.py'))
def test_morphology_seam():
 M['main']();d=json.loads((R/'runs/relative-instrument-morphology-seam-20260921.json').read_text());assert d['candidate_count']==2 and d['exact_count']==0 and d['stats']['longest_letters']>38
 for x in d['rendered_candidates']:assert x['live_trace']['alternating_numbers'] and x['audit']['forward_sha256']!=x['audit']['reverse_sha256']
