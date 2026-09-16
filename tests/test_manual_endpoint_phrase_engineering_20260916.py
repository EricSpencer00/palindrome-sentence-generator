import json
from pathlib import Path
def test_manual_endpoint_run():
 p=json.loads((Path(__file__).parents[1]/'runs/manual-endpoint-engineering-20260916.json').read_text())
 assert p['registry_preflight']['exact_signature_collisions']==[]
 assert p['exact_count']==0 and p['near_miss_count']==18
 assert all(not x['reader_eligible'] for x in p['candidates'])
