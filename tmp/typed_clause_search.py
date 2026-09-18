import itertools, re, time
from wordfreq import zipf_frequency

B = {
 'DET':'a an the some one our my no his her'.split(),
 'HUMAN':'aide artist baker carpenter captain child clerk doctor driver farmer friend gardener guide keeper man master men nurse pilot poet porter reader sailor scholar singer smith soldier student teacher woman writer agent angel author actor elder monk mother father sister brother crew team people girls boys diana maria nora lena anna iris otto ada eve noah olivia uma sara maya'.split(),
 'OBJ':'aid apple atlas boat book bread bridge chart chair candle car case cave coin cove deer desk door dream drum flag flower gate garden glass horse island key lamp lantern letter line list map meal memo memos note notes page paper path pen poem pond rope road room sail seal ship sign song stone story table tent tool toy train tree vase wall water wheel window yard action answer idea image message mission number plan report signal task truth word words'.split(),
 'TV':'aids asks bakes calls carries charts checks cleans cooks cuts draws finds folds gives guides guards helps holds inspire inspires keeps leads likes maps marks mends meets moves needs opens packs paints plants reads repairs rides rips sends sets shares shows sings speaks takes teaches tells trains uses visits walks watches waters writes reviews sees knows loves makes brings builds saves names follows crosses turns returns notices wants seeks'.split(),
 'IMP':'deliver reward draw emit repay repaid repair read write mark guide guard carry take give make see use live stop start'.split(),
 'MOD':'one two three four five six seven eight nine ten old new red blue green small large quiet fresh warm cool kind clear young wise bright dark long short soft hard fine fair true good last first many some more each odd even'.split(),
 'ADJ':'metallic stressed reviled quiet patient gentle young old green bright small kind clear warm fresh calm brave little open early silent red blue wise swift plain honest good dark long short round clean soft great new local wild fine thin white deep high light cool safe true full fair rich sweet heavy'.split(),
 'PROPN':'ada anna diana eve iris lena lisa maria maya nora noah olivia otto sara uma alan eric emil ella ian ava art leo susan ruth james jane john mary paul peter rose mark anne clara david nina tara tina lori lara sara sam ben tom ray eli'.split(),
}
for k in B:B[k]=list(dict.fromkeys(B[k]))
def tape(s):return ''.join(c for c in s if c.isalpha())
CLAUSES=[
 ['DET','HUMAN','TV','MOD','OBJ'],
 ['DET','HUMAN','TV','DET','OBJ'],
 ['DET','HUMAN','TV','PROPN'],
 ['HUMAN','TV','DET','OBJ'],
 ['DET','HUMAN','TV','DET','MOD','OBJ'],
 ['IMP','ADJ','OBJ'],
 ['IMP','OBJ','ADJ'],
 ['DET','OBJ','TV','OBJ'],
]
def search(T,limit=600000):
 sol=[];nodes=0
 def dfs(li,ri,left,right,lw,rw,score):
  nonlocal nodes;nodes+=1
  if nodes>limit:return
  rr=right[::-1];m=min(len(left),len(rr))
  if left[:m]!=rr[:m]:return
  if li>ri:
   if left==rr:sol.append((' '.join(lw+rw),score))
   return
  if li==ri:
   kind=T[li]
   for w in B[kind]:
    if kind in {'HUMAN','OBJ','TV','MOD','PROPN'} and w in lw+rw:continue
    full=left+tape(w)+right
    if full==full[::-1]:sol.append((' '.join(lw+[w]+rw),score+zipf_frequency(w,'en')))
   return
  sides=[]
  if len(left)<=len(rr):sides.append('L')
  if len(rr)<=len(left):sides.append('R')
  if len(sides)==2:sides.sort(key=lambda s:len(B[T[li if s=='L' else ri]]))
  for side in sides:
   kind=T[li if side=='L' else ri]
   for w in B[kind]:
    if kind in {'HUMAN','OBJ','TV','MOD','PROPN'} and w in lw+rw:continue
    if side=='L':
     nl=left+tape(w);m=min(len(nl),len(rr))
     if nl[:m]!=rr[:m]:continue
     dfs(li+1,ri,nl,right,lw+[w],rw,score+zipf_frequency(w,'en'))
    else:
     nr=tape(w)+right;nrr=nr[::-1];m=min(len(left),len(nrr))
     if left[:m]!=nrr[:m]:continue
     dfs(li,ri-1,left,nr,lw,[w]+rw,score+zipf_frequency(w,'en'))
 dfs(0,len(T)-1,'','',[],[],0.)
 return nodes,sol
if __name__=='__main__':
 hits=[];st=time.time()
 for arity in (2,3):
  for choices in itertools.product(range(len(CLAUSES)),repeat=arity):
   T=[]
   for i in choices:T+=CLAUSES[i]
   if len(T)>16:continue
   n,s=search(T)
   for text,sc in s:hits.append((len(tape(text)),sc,text,choices,n))
   if s:print('hit',choices,n,len(s),s[:3],flush=True)
 print('done',len(hits),'sec',time.time()-st)
 for row in sorted(hits,reverse=True)[:100]:print(row)
