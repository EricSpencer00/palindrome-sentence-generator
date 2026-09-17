import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/finite_reverse_phrase_composition_bank_20260916.py')],check=True)
 d=json.loads((R/'runs/finite-reverse-phrase-composition-bank-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>38
 for x in d['rows']:
  assert x['composition_state']['online_obligation'];assert x['anti_shortcut']['finished_tape_resegmentation'] is False;assert x['audit']['sha256_equal'] is False;assert x['next_repair']
