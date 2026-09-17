import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/scalable_outsidein_paired_terminal_repair_20260916.py')],check=True)
 d=json.loads((R/'runs/scalable-outsidein-paired-terminal-repair-20260916.json').read_text());assert d['novelty_preflight']['passed'];assert d['summary']['max_length']>38
 for x in d['rows']:
  assert x['repair_state']['same_role'];assert x['repair_state']['scene_frame_preserved'];assert x['audit']['sha256_equal'] is False;assert x['anti_shortcut']['nested_palindrome_spans'] is False;assert x['next_repair']
