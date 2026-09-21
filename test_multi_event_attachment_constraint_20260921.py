from multi_event_attachment_constraint_20260921 import stream

def test_partial_frontier_survives_first_check():
    ok, checks, frontier = stream('ab', 'ba')
    assert frontier > 1
    assert checks >= 1

def test_global_endpoint_index_exact_and_control():
    assert stream('ab', 'ba')[0]
    assert not stream('ab', 'ca')[0]
