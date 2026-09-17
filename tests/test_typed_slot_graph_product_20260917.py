from experiments.typed_slot_graph_product_20260917 import compile_slots, run, SLOTS
from experiments.exact_palindrome_graph_product_20260917 import CharacterGraph
def test_typed_graph_has_character_edges_and_boundary_provenance():
    g=CharacterGraph.from_phrases(['the calm man sees a dog'],'slot')
    assert any(e.char=='c' for es in g.edges.values() for e in es)
    assert any(e.char is None for es in g.edges.values() for e in es)
def test_typed_lane_reports_honest_zero_and_budget():
    x=run(); assert x['status']=='completed_exact_zero'; assert x['config']['state_budget']>0
    assert all(r['result']['completions']==[] for r in x['shapes'])

def test_slot_graph_rejoins_alternatives_without_phrase_cartesian_product():
    g = compile_slots(('DET', 'PERSON'), provenance='test')
    # Two DET and four PERSON alternatives are represented as branches in a
    # layered graph; no accepting sentence list is stored on the graph.
    assert len(g.accepting) == 1
    assert not g.accepting_paths
    assert len(g.edges[g.start]) == len(SLOTS['DET'])
    assert sum(e.char is None for es in g.edges.values() for e in es) >= len(SLOTS['DET'])

def test_slot_run_records_graph_provenance_and_no_rendered_shortcut():
    x = run()
    for row in x['shapes']:
        assert row['graph_nodes']['left'] > 1
        assert row['graph_nodes']['right'] > 1
        assert row['rendered'] == []
