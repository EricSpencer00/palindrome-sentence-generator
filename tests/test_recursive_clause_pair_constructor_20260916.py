import json
from pathlib import Path
def test_recursive_clause_pair_run():
 p=json.loads((Path(__file__).parents[1]/'runs/recursive-clause-pair-20260916.json').read_text())
 assert p['registry_preflight']['exact_signature_collisions']==[]
 assert p['exact_count']==0 and p['reader_eligible_count']==0
 assert all(x['no_repeated_clauses'] for x in p['candidates'])
