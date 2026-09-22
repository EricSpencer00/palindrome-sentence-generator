from experiments.fourteen_event_triple_cut_20260921 import support
def test_endpoint_pairing():
 assert support(['a.']*14)[0]
 assert not support(['a.']*13+['b.'])[0]
