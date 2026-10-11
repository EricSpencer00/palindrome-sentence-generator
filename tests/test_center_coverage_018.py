import itertools,random
from llm_palindrome.bilateral_seams import Chunk,AllChunkCentersGrammar,OwedSideBilateralGrammar,WholeChunkCenterGrammar,norm

def test_internal_centers_preserve_whole_chunks():
 for l,r in [('We panic ','in a pew.'),('Do geese ','see God?'),('Live not ','on evil.')]:
  cs=[Chunk('l',l,'START','M','predicate','known control'),Chunk('r',r,'M','END','complement','known control')]
  result=AllChunkCentersGrammar(cs).search(max_steps=2)
  assert {o['tape'] for o in result['outputs']}=={norm(l+r)}
  o=result['outputs'][0];assert o['text']==l+r and o['center_ids']
  assert not AllChunkCentersGrammar(cs).search(max_steps=1)['outputs']
  assert len(o['left_ids'])+len(o['right_ids'])+len(o['center_ids'])==2

def test_outer_palindromic_residual_is_not_centerout_closure():
 c=Chunk('x','ab','START','END','formal','diagnostic')
 g=WholeChunkCenterGrammar([c],'x',0,1)
 assert g.initial()['debt']['letters']=='b'
 assert not g.closed(g.initial()) and not g.search()['outputs']

def test_shared_budget_and_soft_rendered_repetition():
 cs=[Chunk('a','aa ','START','M','formal','diagnostic'),Chunk('b','aa','M','END','formal','diagnostic')]
 r=AllChunkCentersGrammar(cs,max_seeds=1).search(max_steps=2)
 assert r['receipt']['seed_status']=='seed_truncated'
 full=AllChunkCentersGrammar(cs).search(max_steps=2)['outputs'][0]
 assert full['repetition']['repeated_tokens']=={'aa':2}
 assert full['repetition']['repeated_phrases']=={'aa':2}

def test_finite_complete_against_exhaustive_oracle():
 rng=random.Random(19027);tested=exact=0
 for case in range(40):
  levels=1+case%4;cs=[];vocab=[]
  for level in range(levels):
   words=sorted({''.join(rng.choice('abc') for _ in range(rng.randrange(1,5))) for _ in range(4)})
   vocab.append(words)
   for i,w in enumerate(words):cs.append(Chunk(f'{level}-{i}',w+' ','START' if level==0 else str(level),'END' if level==levels-1 else str(level+1),'formal','random finite grammar'))
  tapes={''.join(p) for p in itertools.product(*vocab)};expected={t for t in tapes if t==t[::-1]}
  for cls in (AllChunkCentersGrammar,OwedSideBilateralGrammar):
   r=cls(cs).search(max_steps=levels,max_states=10000,max_outputs=1000,seconds=5)
   assert r['receipt']['status']=='complete_bounded_depth'
   assert {o['tape'] for o in r['outputs']}==expected
  tested+=len(tapes);exact+=len(expected)
 assert (tested,exact)==(2994,68)
