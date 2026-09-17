import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/lexical_mirror_hypergraph_clause_20260916.py')],check=True)
 d=json.loads((R/'runs/lexical-mirror-hypergraph-clause-20260916.json').read_text());x=d['rows'][0];assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>38;assert x['hypergraph']['connected_roles'];assert x['audit']['sha256_equal'] is False;assert x['next_repair']
