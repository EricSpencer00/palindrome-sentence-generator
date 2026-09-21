import importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location("v",R/"experiments/character_wfsa_valency_20260920.py");v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
def test_valency_run():
 o=v.run();assert o["novelty_preflight"]["status"]=="passed";assert o["stats"]["prose_controls"]==4;assert o["config"]["typed_features"]==["number","valency"]
def test_audit(): assert v.audit("The sailor waits") ["letters"]==14
