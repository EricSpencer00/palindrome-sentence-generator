import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('lane',ROOT/'experiments/immutable_scene_exact_tape_resegment_20260916.py');lane=importlib.util.module_from_spec(spec);spec.loader.exec_module(lane)
def test_immutable_tape_diagnostic():
 lane.main();d=json.loads((ROOT/'runs'/(lane.ID+'.json')).read_text())
 assert d['candidate']['complete_prose'] and d['immutable_tape']['frozen']
 assert d['candidate']['audit']['two_pointer_exact'] is False and d['resegmentation']['sweep'] is False
 assert d['novelty_preflight']['catalogue_text_imported'] is False
