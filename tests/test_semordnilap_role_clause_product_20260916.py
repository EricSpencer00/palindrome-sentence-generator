import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/semordnilap_role_clause_product_20260916.py')],check=True)
 d=json.loads((R/'runs/semordnilap-role-clause-product-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>100
 for x in d['rows']:
  assert x['audit']['sha256_equal'] is False;assert x['audit']['independent_two_pointer_exact'] is False;assert x['anti_shortcut']['semordnilap_unit_repeated'] is False;assert x['next_repair']
