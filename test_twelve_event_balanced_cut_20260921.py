from twelve_event_balanced_cut_20260921 import support
def test_endpoint_pairing():
 assert support(['a.']*12)[0]
 assert not support(['a.']*11+['b.'])[0]
