import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/boundary_shift_semordnilap_grammar_20260916.py')],check=True)
 d=json.loads((R/'runs/boundary-shift-semordnilap-grammar-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>100
 for x in d['rows']:
  assert x['audit']['sha256_equal'] is False;assert x['audit']['independent_two_pointer_exact'] is False;assert x['anti_shortcut']['posthoc_reverse_rendering'] is False;assert x['next_repair']
