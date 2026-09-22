from experiments.thirteen_event_dual_cut_20260921 import support
def test_endpoint_pairing():
 assert support(['a.']*13)[0]
 assert not support(['a.']*12+['b.'])[0]
