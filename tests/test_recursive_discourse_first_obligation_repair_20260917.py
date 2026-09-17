import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('m',Path(__file__).parents[1]/'experiments/recursive_discourse_first_obligation_repair_20260917.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_targeted_repair_is_recorded_and_audited():
 r=m.run();assert r['config']['targeted_first_obligation_repair'];assert r['provenance']['condition'];assert r['stats']['rendered']
 assert all(x['independent_exact_audit']['exact'] for x in r['closures'])
