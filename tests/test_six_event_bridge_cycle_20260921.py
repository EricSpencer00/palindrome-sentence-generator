from experiments.six_event_bridge_cycle_20260921 import support
def test_endpoint_pairing():
 assert support(['ab.','x.','x.','x.','x.','ba.'])[0]
 assert not support(['ab.','x.','x.','x.','x.','ca.'])[0]
