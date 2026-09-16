import json
from pathlib import Path
from experiments.authored_clause_template_sat_20260916 import run,SIGNATURE
def test_template_sat_has_real_prose_and_independent_audit():
 x=run(); assert len(x["candidates"])==13; assert x["stats"]["exact"]==0
 assert all(c["provenance"]["authored_templates"] for c in x["candidates"])
 assert all(c["exact_audit"]["hash_forward"]!=c["exact_audit"]["hash_reverse"] for c in x["candidates"])
 assert x["novelty_preflight"]["exact_signature_collision"] is False
def test_artifact():
 x=json.loads(Path("runs/authored-clause-template-sat-20260916.json").read_text()); assert x["signature"]==SIGNATURE; assert x["next_repair"]
