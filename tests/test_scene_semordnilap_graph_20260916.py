from experiments.scene_semordnilap_graph_20260916 import run
def test_scene_graph_keeps_edges_diagnostic_not_shortcut():
 p=run();assert p['stats']['scenes']==2;assert p['novelty_preflight']['status']=='passed'
 for r in p['candidates']:
  assert r['letters']>90;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal'];assert r['provenance']['semordnilap_edge_is_not_presented_as_exact']
