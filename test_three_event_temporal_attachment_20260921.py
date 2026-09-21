from three_event_temporal_attachment_20260921 import support
def test_global_endpoint_pairing():
    assert support(['ab.','x.','ba.'])[0]
    assert not support(['ab.','x.','ca.'])[0]
