"""Six-slot phrase grammar with live two-sided character obligations."""
from __future__ import annotations
import hashlib,json,re
from collections import defaultdict
from pathlib import Path
from nltk.corpus import brown
ROOT=Path(__file__).resolve().parents[1]
def letters(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); return {'letters':len(t),'exact':bool(t) and t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest()}
def banks():
 b=defaultdict(set)
 for sent in brown.tagged_sents():
  for i in range(len(sent)-1):
   (a,ta),(c,tc)=sent[i:i+2]; a=a.lower();c=c.lower()
   if not(re.fullmatch('[a-z]+',a) and re.fullmatch('[a-z]+',c)): continue
   p=a+' '+c
   if ta.startswith(('AT','DT','JJ','NN')) and tc.startswith('NN'): b['NP'].add(p)
   if ta.startswith('VB') and tc.startswith(('AT','DT','NN','JJ')): b['VP'].add(p)
   if ta.startswith('IN') and tc.startswith(('AT','DT','NN','JJ')): b['PP'].add(p)
 return {k:tuple(sorted(v))[:180] for k,v in b.items()}
def run(limit=100000):
 bs=banks(); roles=('NP','VP','PP','PP','VP','NP'); rev={k:defaultdict(list) for k in bs}
 for k,ws in bs.items():
  for w in ws: rev[k][letters(w)[::-1][0]].append(w)
 states=pruned=0; found=[]
 def walk(lo,hi,left,right,words):
  nonlocal states,pruned
  if states>=limit:return
  if lo>hi:
   states+=1; text=' '.join(words)
   if audit(text)['exact'] and len(letters(text))>38: found.append({'rendered':text,'audit':audit(text),'provenance':{'roles':roles,'recombined':True,'reverse_trie_queried':True}})
   return
  for l in bs[roles[lo]]:
   if l in words: continue
   nl0=left+letters(l)
   # If left has residual characters, their first character dictates the
   # right phrase's final character. Otherwise an existing right residual
   # dictates the new left phrase's first character.
   if left:
    right_words=rev[roles[hi]].get(nl0[0],())
   elif right:
    if letters(l)[0] != right[-1]: continue
    right_words=bs[roles[hi]]
   else:
    right_words=bs[roles[hi]]
   for r in right_words:
    if r in words or r==l: continue
    nl=nl0; nr=letters(r)+right; n=min(len(nl),len(nr))
    if nl[:n]!=nr[::-1][:n]: pruned+=1; continue
    walk(lo+1,hi-1,nl[n:],nr[:-n] if n else nr,words+[l,r])
 walk(0,5,'','',[])
 return {'method':'six-slot-phrase-clause-20260919','roles':roles,'bank_sizes':{k:len(v) for k,v in bs.items()},'states':states,'pruned':pruned,'exact_candidates':found,'candidate_count':len(found),'status':'no reader candidate' if not found else 'requires blinded reading'}
if __name__=='__main__':
 o=run();(ROOT/'runs/six-slot-phrase-clause-20260919.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
