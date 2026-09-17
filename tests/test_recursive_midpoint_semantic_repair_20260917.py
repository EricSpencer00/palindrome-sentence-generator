import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('m',Path(__file__).parents[1]/'experiments/recursive_midpoint_semantic_repair_20260917.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_midpoint_repair_tracks_offsets_and_audit():
 r=m.run();assert r['config']['midpoint_crossing_product'];assert r['config']['inside_token_crossing'];assert r['stats']['rendered'];assert all(x['independent_exact_audit']['exact'] for x in r['closures'])
