import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/cfg_earley_fresh_typed_adjunct_repair_20260916.py')],check=True)
 d=json.loads((R/'runs/cfg-earley-fresh-typed-adjunct-repair-20260916.json').read_text());x=d['rows'][0];assert d['novelty_preflight']['passed'];assert x['chart_state']['complete_svo'];assert x['audit']['sha256_equal'] is False;assert x['next_repair']
