import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/cross_boundary_phrase_block_grammar_20260916.py')],check=True)
 d=json.loads((R/'runs/cross-boundary-phrase-block-grammar-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>38
 for x in d['rows']:
  assert x['block_derivation']['mapping'];assert x['anti_shortcut']['non_word_order_mapping'];assert x['audit']['sha256_equal'] is False;assert x['next_repair']
