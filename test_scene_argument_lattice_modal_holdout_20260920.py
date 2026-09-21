import json
from pathlib import Path

def test_run_has_live_audits_and_holdout():
 d=json.loads(Path('runs/scene-argument-lattice-modal-holdout-20260920.json').read_text())
 assert d['novelty_preflight']['status']=='passed'
 assert d['stats']['live_edge_rejections']>0
 assert d['stats']['rendered_candidates']>0
 for row in d['candidates'][:10]:
  assert row['provenance']['held_out_evidence_verbs']
  assert set(('sha256_forward','sha256_reverse','pointer_exact')) <= set(row['audit'])
