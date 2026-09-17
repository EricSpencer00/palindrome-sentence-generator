import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/semantic_phrase_edge_graph_joiner_20260916.py'); lane=importlib.util.module_from_spec(spec); spec.loader.exec_module(lane)
def test_phrase_edges_join_complete_prose():
 lane.main(); d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text())
 assert len(d['candidates'])==3 and all(x['audit']['letters']>100 for x in d['candidates'])
 assert all(x['semantic_consistency'] and x['audit']['two_pointer_exact'] is False for x in d['candidates'])
 assert d['novelty_preflight']['fixed_tape_used'] is False
