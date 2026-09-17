import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/past_tense_dependency_transducer_20260916.py')],check=True)
 d=json.loads((R/'runs/past-tense-dependency-transducer-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>100
 for x in d['rows']:
  assert x['dependency_tree']['agreement'];assert x['anti_shortcut']['no_word_order_symmetry'];assert x['audit']['sha256_equal'] is False;assert x['next_repair']
