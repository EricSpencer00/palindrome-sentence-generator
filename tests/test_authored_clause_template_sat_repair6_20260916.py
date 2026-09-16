import json
from pathlib import Path
from experiments.authored_clause_template_sat_repair6_20260916 import run,SIGNATURE
def test_repair6_is_single_fresh_subject_object_pair():
 x=run(); c=x["candidates"][0]
 assert x["stats"]["candidates"]==1 and x["stats"]["exact"]==0
 assert c["targeted_outer_pair"]["matched"] and c["provenance"]["fresh_subject_object_authored"]
 assert c["fresh_subject_object_pair"]=={"subject":"the patient guide","object":"a curious visitor"}
 assert c["exact_audit"]["hash_forward"]!=c["exact_audit"]["hash_reverse"] and c["letters"]>=39
def test_repair6_artifact_schema():
 x=json.loads(Path("runs/authored-clause-template-sat-repair6-20260916.json").read_text()); assert x["signature"]==SIGNATURE
