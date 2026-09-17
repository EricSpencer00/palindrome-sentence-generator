import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/grammar_first_matching_boundary_run_20260916.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def test_matching_boundaries_then_global_audit():
 lane.main();d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text());c=d['candidate']
 assert c['complete_prose'] and c['audit']['letters']>38 and c['audit']['two_pointer_exact'] is False
 assert all(x['equation_satisfied_before_commit'] for x in c['committed_boundaries'])
 assert d['novelty_preflight']['finished_tape_input'] is False
