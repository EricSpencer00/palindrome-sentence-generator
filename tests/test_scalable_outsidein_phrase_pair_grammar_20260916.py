import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/scalable_outsidein_phrase_pair_grammar_20260916.py')],check=True)
 d=json.loads((R/'runs/scalable-outsidein-phrase-pair-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>38
 for x in d['rows']:
  assert x['outsidein_state']['joint_terminal_rule'];assert x['anti_shortcut']['nested_palindrome_spans'] is False;assert x['anti_shortcut']['seed_wrapping'] is False;assert x['audit']['sha256_equal'] is False;assert x['next_repair']
