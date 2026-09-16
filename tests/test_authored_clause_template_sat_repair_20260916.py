import json
from pathlib import Path
from experiments.authored_clause_template_sat_repair_20260916 import run,SIGNATURE
def test_repair_is_one_fresh_complete_frame_with_independent_audit():
 x=run(); c=x["candidates"][0]
 assert x["stats"]["candidates"]==1 and x["stats"]["exact"]==0
 assert c["provenance"]["fresh_frame_authored"] and c["source_first_mismatch_index"]>=0
 assert c["exact_audit"]["hash_forward"]!=c["exact_audit"]["hash_reverse"]
 assert c["letters"]>=39 and x["next_repair"]
def test_repair_artifact_schema():
 x=json.loads(Path("runs/authored-clause-template-sat-repair-20260916.json").read_text()); assert x["signature"]==SIGNATURE
