from eleven_event_asymmetric_cut_20260921 import support
def test_endpoint_pairing():
 assert support(['a.']*11)[0]
 assert not support(['a.']*10+['b.'])[0]
