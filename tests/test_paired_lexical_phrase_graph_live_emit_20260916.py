import json,subprocess,sys
from pathlib import Path
R=Path(__file__).parents[1]
def test_contract():
 subprocess.run([sys.executable,str(R/'experiments/paired_lexical_phrase_graph_live_emit_20260916.py')],check=True)
 d=json.loads((R/'runs/paired-lexical-phrase-graph-live-emit-20260916.json').read_text());x=d['rows'][0];assert d['novelty_preflight']['passed'];assert x['graph_state']['emission'];assert x['audit']['sha256_equal'] is False;assert x['next_repair']
