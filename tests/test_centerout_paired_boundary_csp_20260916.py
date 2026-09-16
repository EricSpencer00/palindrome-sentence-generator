import json
from pathlib import Path
from experiments.centerout_paired_boundary_csp_20260916 import run, EXPERIMENT_ID
def test_paired_boundary_csp():
 p=run();r=p['candidate'];assert p['stats']['bounded_assignments']==4;assert p['stats']['candidates']==1;assert r['exact_audit']['two_pointer_exact'] is False;assert r['exact_audit']['sha_equal'] is False;assert r['provenance']['ordinary_svo_order'];assert r['letters']>=39;reg=json.loads((Path(__file__).resolve().parents[1]/'docs/experiment-novelty-registry.json').read_text());assert any(x['id']==EXPERIMENT_ID for x in reg['entries'])
