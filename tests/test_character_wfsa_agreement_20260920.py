import importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[1]; s=importlib.util.spec_from_file_location("x",R/"experiments/character_wfsa_agreement_20260920.py"); x=importlib.util.module_from_spec(s); s.loader.exec_module(x)
def test_typed_agreement_run():
 o=x.run(); assert o["novelty_preflight"]["status"]=="passed"; assert o["stats"]["prose_controls"]==3; assert o["config"]["typed_features"]==["number"]
def test_independent_audit():
 a=x.audit("The sailor guides a harbor"); assert a["letters"]==22 and not a["pointer_exact"]
