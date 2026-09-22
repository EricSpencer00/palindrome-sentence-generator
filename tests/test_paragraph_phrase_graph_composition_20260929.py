from experiments.paragraph_phrase_graph_composition_20260929 import run, letters

def test_new_outer_edge_is_exact_and_longer():
    item = run()["rendered_candidates"][0]
    assert item["audit"]["two_pointer_exact"]
    assert item["audit"]["sha_equal"]
    assert item["audit"]["letters"] > 156
    assert item["provenance"]["streamed_against_opposing_obligations"]

def test_independent_normalization():
    text = run()["rendered_candidates"][0]["text"]
    tape = letters(text)
    assert tape == tape[::-1]
