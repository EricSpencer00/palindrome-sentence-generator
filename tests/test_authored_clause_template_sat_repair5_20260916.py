import json
from pathlib import Path
from experiments.authored_clause_template_sat_repair5_20260916 import run,SIGNATURE
def test_repair5_is_single_fresh_verb_frame():
 x=run(); c=x["candidates"][0]
 assert x["stats"]["candidates"]==1 and x["stats"]["exact"]==0
 assert c["targeted_outer_pair"]["matched"] and c["fresh_guide_verb"]=="leads"
 assert c["provenance"]["preserved_roles"]==["usher","guide","visitor"]
 assert c["exact_audit"]["hash_forward"]!=c["exact_audit"]["hash_reverse"] and c["letters"]>=39
def test_repair5_artifact_schema():
 x=json.loads(Path("runs/authored-clause-template-sat-repair5-20260916.json").read_text()); assert x["signature"]==SIGNATURE
