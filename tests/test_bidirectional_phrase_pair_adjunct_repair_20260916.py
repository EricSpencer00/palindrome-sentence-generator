import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/bidirectional_phrase_pair_adjunct_repair_20260916.py')],check=True)
 d=json.loads((R/'runs/bidirectional-phrase-pair-adjunct-repair-20260916.json').read_text());x=d['rows'][0];assert d['summary']['max_length']>38;assert x['repair_state']['single_child'];assert x['audit']['sha256_equal'] is False;assert x['next_repair']
