from experiments.semantic_wordpair_event_graph_20260916 import run
def test_wordpair_graph_keeps_svo_and_rejects_chains():
 p=run();assert p['stats']['bounded_paths']==1;assert p['novelty_preflight']['status']=='passed';r=p['candidate'];assert r['provenance']['semordnilap_chain_rejected'];assert r['letters']>38;assert not r['exact_audit']['two_pointer_exact'];assert not r['exact_audit']['sha_equal']
