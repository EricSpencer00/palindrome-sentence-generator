from experiments.typed_slot_graph_product_20260917 import run
from experiments.exact_palindrome_graph_product_20260917 import CharacterGraph
def test_typed_graph_has_character_edges_and_boundary_provenance():
    g=CharacterGraph.from_phrases(['the calm man sees a dog'],'slot')
    assert any(e.char=='c' for es in g.edges.values() for e in es)
    assert any(e.char is None for es in g.edges.values() for e in es)
def test_typed_lane_reports_honest_zero_and_budget():
    x=run(); assert x['status']=='completed_exact_zero'; assert x['config']['state_budget']>0
    assert all(r['result']['completions']==[] for r in x['shapes'])
