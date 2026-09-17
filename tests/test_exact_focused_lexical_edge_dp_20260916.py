import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/exact_focused_lexical_edge_dp_20260916.py')],check=True)
 d=json.loads((R/'runs/exact-focused-lexical-edge-dp-20260916.json').read_text());x=d['rows'][0];assert d['novelty_preflight']['status']=='excluded_duplicate';assert d['novelty_preflight']['duplicate_rendered_candidate'] is True;assert x['dp_state']['posthoc_filtering'] is False;assert x['audit']['sha256_equal'] is False;assert x['next_repair']
