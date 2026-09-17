import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/brown_phrase_pair_seam_20260916.py')],check=True)
 d=json.loads((R/'runs/brown-phrase-pair-seam-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>38
 for x in d['rows']:
  assert x['phrase_channel']['cross_word_obligation'];assert x['audit']['sha256_equal'] is False;assert x['audit']['independent_two_pointer_exact'] is False;assert x['anti_shortcut']['finished_sentence_reversal'] is False;assert x['next_repair']
