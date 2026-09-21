from five_event_cycle_attachment_20260921 import support
def test_endpoint_pairing():
 assert support(['ab.','x.','x.','x.','ba.'])[0]
 assert not support(['ab.','x.','x.','x.','ca.'])[0]
