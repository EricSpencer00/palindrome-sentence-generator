import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/live_tape_clause_terminal_decoder_20260916.py')],check=True)
 d=json.loads((R/'runs/live-tape-clause-terminal-decoder-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>100
 for x in d['rows']:
  assert x['decoder_state']['mirrored_tape_obligation']=='live at each terminal';assert x['audit']['sha256_equal'] is False;assert x['audit']['independent_two_pointer_exact'] is False;assert x['next_repair']
