import itertools, re, time
from collections import defaultdict
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

def clean(x): return x.lower().replace('_','')
def wn_bank(lexnames, pos='n', limit=500):
    vals=set()
    for syn in wn.all_synsets(pos):
        if syn.lexname() not in lexnames: continue
        for lemma in syn.lemma_names():
            w=clean(lemma)
            if w.isalpha() and w.isascii() and len(w)>=3 and zipf_frequency(w,'en')>=3.0:
                vals.add(w)
    return sorted(vals,key=lambda w:(-zipf_frequency(w,'en'),w))[:limit]

B={
 'DET':'a an the some one our my no his her'.split(),
 'HUMAN':wn_bank({'noun.person','noun.group'},limit=500),
 'OBJ':wn_bank({'noun.artifact','noun.object','noun.food','noun.plant','noun.location','noun.communication'},limit=800),
 'ADJ':wn_bank({'adj.all'},pos='a',limit=600),
 'TV':[],
 'MOD':'one two three four five six seven eight nine ten old new red blue green small large quiet fresh warm cool kind clear young wise bright dark long short soft hard fine fair true good last first many some more each odd even'.split(),
 'PROPN':'ada anna diana eve iris lena lisa maria maya nora noah olivia otto sara uma alan eric emil ella ian ava art leo susan ruth james jane john mary paul peter rose mark anne clara david nina tara tina lori lara sara sam ben tom ray eli'.split(),
}
# Keep ordinary transitive/action verbs from a curated surface inventory; use
# WordNet to expand with common lexicalized forms that Brown tags as verbs.
for syn in wn.all_synsets('v'):
    if syn.lexname() not in {'verb.contact','verb.communication','verb.creation','verb.consumption','verb.possession','verb.social','verb.motion','verb.cognition','verb.change'}: continue
    for lemma in syn.lemma_names():
        w=clean(lemma)
        if w.isalpha() and w.isascii() and len(w)>=3 and zipf_frequency(w,'en')>=3.5:
            B['TV'].append(w)
B['TV'] += 'aids asks bakes calls carries charts checks cleans cooks cuts draws finds folds gives guides guards helps holds inspire inspires keeps leads likes maps marks mends meets moves needs opens packs paints plants reads repairs rides rips sends sets shares shows sings speaks takes teaches tells trains uses visits walks watches waters writes reviews sees knows loves makes brings builds saves names follows crosses turns returns notices wants seeks'.split()
for k in B:B[k]=list(dict.fromkeys(B[k]))
print({k:len(v) for k,v in B.items()})

def tape(w):return ''.join(c for c in w if c.isalpha())

# Prefix index so the live obligation lookup does not scan an entire lexical bank.
IDX={k:defaultdict(list) for k in B}
for k,vals in B.items():
 for w in vals:IDX[k][tape(w)[:1]].append(w)

def search(T,limit=500000):
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
   k=T[li]
   for w in B[k]:
    if k in {'HUMAN','OBJ','TV','ADJ','MOD','PROPN'} and w in lw+rw:continue
    full=left+tape(w)+right
    if full==full[::-1]:sol.append((' '.join(lw+[w]+rw),score+zipf_frequency(w,'en')))
   return
  sides=[]
  if len(left)<=len(rr):sides.append('L')
  if len(rr)<=len(left):sides.append('R')
  if len(sides)==2:sides.sort(key=lambda s:len(B[T[li if s=='L' else ri]]))
  debtL=rr[len(left):] if len(rr)>len(left) else ''
  debtR=left[len(rr):] if len(left)>len(rr) else ''
  for side in sides:
   k=T[li if side=='L' else ri]
   if not (debtL or debtR):
    cand = B[k]
   elif side == 'L':
    cand = [w for w in B[k] if tape(w).startswith(debtL) or debtL.startswith(tape(w))]
   else:
    cand = [w for w in B[k] if tape(w)[::-1].startswith(debtR) or debtR.startswith(tape(w)[::-1])]
   for w in cand:
    if k in {'HUMAN','OBJ','TV','ADJ','MOD','PROPN'} and w in lw+rw:continue
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

CLAUSES=[
 ['DET','HUMAN','TV','MOD','OBJ'],
 ['DET','HUMAN','TV','DET','OBJ'],
 ['DET','HUMAN','TV','PROPN'],
 ['HUMAN','TV','DET','OBJ'],
 ['DET','HUMAN','TV','DET','MOD','OBJ'],
 ['DET','ADJ','HUMAN','TV','DET','OBJ'],
 ['DET','HUMAN','TV','DET','ADJ','OBJ'],
]
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
