import json,runpy
from pathlib import Path
R=Path(__file__).resolve().parents[1];M=runpy.run_path(str(R/'experiments/semantic_branch_spatial_scene_20260921.py'))
def test_spatial_scene():
 M['main']();d=json.loads((R/'runs/semantic-branch-spatial-scene-20260921.json').read_text());assert d['candidate_count']==2 and d['exact_count']==0 and d['novelty_preflight']['all_rendered_new'] and d['stats']['longest_letters']>38
 for x in d['rendered_candidates']:assert x['semantic_graph']['shared_spatial_scene_after_convergence'] and x['audit']['forward_sha256']!=x['audit']['reverse_sha256']
