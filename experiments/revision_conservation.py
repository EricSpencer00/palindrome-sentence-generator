"""Vocabulary/traversal sensitivity of enumerated-edge statistics, not trajectories."""
import collections,json,statistics,sys,time
sys.path.insert(0,'.')
from llm_palindrome.generate import build_vocab
from llm_palindrome.search import WordTries,consume,unit_letters
from llm_palindrome.centerout import consume_suffix
rows=[]
for requested in [1000,2000,4000,6000,8000,16000,32000,47000]:
 vocab=build_vocab(requested);tries=WordTries(vocab)
 for mode in ['breadth_first','depth_first']:
  q=collections.deque(['']);seen={''};net=[];settled=[];units=[];start=time.monotonic()
  while q and len(net)<60000:
   o=q.popleft() if mode=='breadth_first' else q.pop()
   for cands,take in [(tries.left_candidates(o,400),consume),(tries.right_candidates(o[::-1],400),consume_suffix)]:
    for w in cands:
     s=unit_letters(w);res=take(s,o)
     if res is None:continue
     new,_=res;net.append(len(o)-len(new));settled.append(min(len(o),len(s)));units.append(len(s))
     assert abs(len(o)-len(s))==len(new)
     if len(net)>=60000:break
     if new not in seen and len(new)<=24:seen.add(new);q.append(new)
    if len(net)>=60000:break
  rows.append(dict(requested=requested,actual=len(vocab),mode=mode,n=len(net),states=len(seen),mean_net=statistics.mean(net),mean_settled=statistics.mean(settled),mean_unit=statistics.mean(units),seconds=time.monotonic()-start))
  print(rows[-1],flush=True)
  open('runs/revision-2026-09-07/conservation.json','w').write(json.dumps(rows,indent=2)+'\n')
