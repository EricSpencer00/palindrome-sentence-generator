import json
from experiments.scene_seam_csp_boundary_20260917 import OUT,main
def test_live_scene_seam_report():
 main(); p=json.loads(OUT.read_text()); assert p["novelty_preflight"]["passed"]
 assert p["summary"]["products"]==17 and p["summary"]["exact"]==0
 repair=[r for r in p["rows"] if "recorded the data" in r["rendered"]][0]
 assert repair["live_seam"]["trace"][0]["matched"] >= 2
 r=p["best_frontier"]; assert r["audit"]["two_pointer_exact"]==r["audit"]["reverse_slice_exact"]
 assert r["provenance"]["word_boundary_crossing"] and not r["provenance"]["mechanically_admitted"]
