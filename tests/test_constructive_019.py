import itertools
from experiments.construct_geese_canoe_019 import make_inventory,MATERIALS,USES,TARGET
from llm_palindrome.bilateral_seams import AllChunkCentersGrammar,OwedSideBilateralGrammar,WholeChunkCenterGrammar,norm

def test_whole_chunk_canoe_extension_and_complete_bounded_grid():
 cs=make_inventory();assert len(cs)==14 and all(len(c.text.split())>=2 for c in cs)
 paths=[f'Do geese on a {m} {u} canoe see God?' for m,u in itertools.product(MATERIALS,USES)]
 exact={norm(t) for t in paths if norm(t)==norm(t)[::-1]};assert len(paths)==36 and exact=={norm(TARGET)}
 for cls in (AllChunkCentersGrammar,OwedSideBilateralGrammar):
  r=cls(cs).search(max_steps=4,max_states=10000,max_outputs=100,seconds=2)
  assert r['receipt']['status']=='complete_bounded_depth'
  assert {o['tape'] for o in r['outputs']}==exact
  assert r['outputs'][0]['repetition']['repeated_tokens']=={}
 g=WholeChunkCenterGrammar(cs,'R2-trade',0,1);s=g.initial();by={c.id:c for c in cs};ratios=[]
 for cid,side in [('L2-cedar','left'),('L1','left'),('R1','right')]:
  s,e=g.extend(s,by[cid],side);assert e is None;ratios.append(s['trace'][-1]['payoff']['ratio'])
 assert ratios==[1,1/7,1] and g.closed(s) and g.render(s)==TARGET
 assert len(norm(TARGET))==31 and norm('on a cedar trade canoe')!=norm('on a cedar trade canoe')[::-1]
