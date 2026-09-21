from four_event_temporal_attachment_20260921 import support
def test_endpoint_pairing():
 assert support(['ab.','x.','x.','ba.'])[0]
 assert not support(['ab.','x.','y.','ca.'])[0]
