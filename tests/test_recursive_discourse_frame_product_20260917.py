import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('m',Path(__file__).parents[1]/'experiments/recursive_discourse_frame_product_20260917.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_recursive_product_has_provenance_and_audit():
 r=m.run();assert r['config']['recursive_semantic_frames'];assert r['stats']['rendered']
 assert all(x['independent_exact_audit']['exact'] for x in r['closures'])
 assert all(x['provenance']['unique_roles'] for x in r['diagnostic_witnesses'])
