import json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];M=runpy.run_path(str(R/'experiments/relative_instrument_location_20260921.py'))
def test_typed_family():
 M['main']();d=json.loads((R/'runs/relative-instrument-location-20260921.json').read_text());assert d['candidate_count']==5 and d['exact_count']==0 and d['stats']['longest_letters']>38
 for x in d['rendered_candidates']:assert x['audit']['forward_sha256']!=x['audit']['reverse_sha256']
