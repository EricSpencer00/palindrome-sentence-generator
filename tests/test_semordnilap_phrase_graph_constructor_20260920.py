from semordnilap_phrase_graph_constructor_20260920 import role, run

def test_phrase_roles_reject_fragments_and_accept_noun_phrases():
    assert role('as time') == 'PP'
    assert role('emits a') == 'VP'
    assert role('the sailor') == 'NP'

def test_only_typed_edges_are_rendered():
    result = run()
    assert result['stats']['rendered_candidates'] == result['stats']['usable_graph_edges'] * 9
    assert all(' answers emits a' not in row['rendered'] for row in result['rendered_candidates'])
