import json
from pathlib import Path
def test_operator():
 d=json.loads(Path('runs/scene-argument-lattice-contrastive-switch-20260920.json').read_text())
 assert d['novelty_preflight']['status']=='passed' and d['stats']['live_edge_rejections']>0
 assert d['stats']['rendered_candidates']>0
 for x in d['candidates'][:10]:
  assert x['provenance']['held_out_evidence_nouns'] and x['provenance']['explicit_agent_switch']
  assert x['live_edges']['agent_switch']
