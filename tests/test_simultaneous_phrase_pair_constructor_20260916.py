import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/simultaneous_phrase_pair_constructor_20260916.py')],check=True)
 d=json.loads((R/'runs/simultaneous-phrase-pair-constructor-20260916.json').read_text());x=d['rows'][0];assert d['novelty_preflight']['passed'];assert x['constructor_state']['phrase_lengths']=='jointly selected';assert x['audit']['sha256_equal'] is False;assert x['next_repair']
