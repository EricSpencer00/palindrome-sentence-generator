import json
from pathlib import Path
from experiments.authored_clause_template_sat_repair7_20260916 import run,SIGNATURE
def test_repair7_is_single_complete_scene_with_new_recipient():
 x=run(); c=x["candidates"][0]
 assert x["stats"]["candidates"]==1 and x["stats"]["exact"]==0
 assert c["targeted_outer_pair"]["matched"] and c["recipient_role"]=="child"
 assert c["provenance"]["fresh_scene_authored"] and c["provenance"]["preserved_valency"]
 assert c["exact_audit"]["hash_forward"]!=c["exact_audit"]["hash_reverse"] and c["letters"]>=39
def test_repair7_artifact_schema():
 x=json.loads(Path("runs/authored-clause-template-sat-repair7-20260916.json").read_text()); assert x["signature"]==SIGNATURE
