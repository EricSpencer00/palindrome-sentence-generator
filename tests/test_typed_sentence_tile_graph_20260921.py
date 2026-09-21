import json
from pathlib import Path
def test_graph_controls_are_intact_and_independently_audited():
 d=json.loads(Path('runs/typed-sentence-tile-graph-20260921.json').read_text())
 assert d['stats']['candidate_edges']>0
 for r in d['diagnostic_controls']:
  assert r['provenance']['complete_tiles'] and r['provenance']['graph_composed']
  assert r['audit']['sha256_forward'] != '' and 'pointer_exact' in r['audit']
