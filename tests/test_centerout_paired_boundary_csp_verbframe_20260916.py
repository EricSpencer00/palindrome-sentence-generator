import json
from pathlib import Path
from experiments.centerout_paired_boundary_csp_verbframe_20260916 import run, EXPERIMENT_ID
def test_fresh_verbframe_csp():
 p=run();r=p['candidate'];assert p['stats']=={'bounded_assignments':8,'candidates':1,'exact':0,'mechanically_admitted':0};assert r['provenance']['fresh_lexical_domains'];assert r['provenance']['ordinary_svo_order'];assert r['exact_audit']['two_pointer_exact'] is False;assert r['exact_audit']['sha_equal'] is False;assert r['letters']>=39;reg=json.loads((Path(__file__).resolve().parents[1]/'docs/experiment-novelty-registry.json').read_text());assert any(x['id']==EXPERIMENT_ID for x in reg['entries'])
