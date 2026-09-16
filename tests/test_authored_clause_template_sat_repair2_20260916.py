import json
from pathlib import Path
from experiments.authored_clause_template_sat_repair2_20260916 import run,SIGNATURE
def test_repair2_targets_outer_character_without_widening():
 x=run(); c=x["candidates"][0]
 assert x["stats"]["candidates"]==1 and x["stats"]["exact"]==0
 assert c["targeted_outer_pair"]["matched"] is True
 assert c["provenance"]["adjunct_only_repair"] and c["exact_audit"]["hash_forward"]!=c["exact_audit"]["hash_reverse"]
 assert c["letters"]>=39 and x["next_repair"]
def test_repair2_artifact_schema():
 x=json.loads(Path("runs/authored-clause-template-sat-repair2-20260916.json").read_text()); assert x["signature"]==SIGNATURE
