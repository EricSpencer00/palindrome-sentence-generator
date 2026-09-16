import json
from pathlib import Path
from experiments.authored_clause_template_sat_repair3_20260916 import run,SIGNATURE
def test_repair3_is_single_fresh_subject_frame():
 x=run(); c=x["candidates"][0]
 assert x["stats"]["candidates"]==1 and x["stats"]["exact"]==0
 assert c["targeted_outer_pair"]["matched"] and c["provenance"]["fresh_subject_authored"]
 assert c["provenance"]["preserved_roles"]==["usher","guide","visitor"]
 assert c["exact_audit"]["hash_forward"]!=c["exact_audit"]["hash_reverse"] and c["letters"]>=39
def test_repair3_artifact_schema():
 x=json.loads(Path("runs/authored-clause-template-sat-repair3-20260916.json").read_text()); assert x["signature"]==SIGNATURE
