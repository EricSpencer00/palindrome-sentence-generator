"""Exact-by-construction NP/VP/PP phrase-clause transducer."""
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
   (w1,t1),(w2,t2)=sent[i:i+2]; w1=w1.lower();w2=w2.lower()
   if not (re.fullmatch('[a-z]+',w1) and re.fullmatch('[a-z]+',w2)): continue
   p=w1+' '+w2
   if t1.startswith(('AT','DT','JJ','NN')) and t2.startswith('NN'): b['NP'].add(p)
   if t1.startswith('VB') and t2.startswith(('AT','DT','NN','JJ')): b['VP'].add(p)
   if t1.startswith('IN') and t2.startswith(('AT','DT','NN','JJ')): b['PP'].add(p)
 return {k:tuple(sorted(v))[:500] for k,v in b.items()}
def consume(a,b):
 n=min(len(a),len(b)); return (a[n:],b[:-n] if n else b) if a[:n]==b[::-1][:n] else None
def run(limit=100000):
 bs=banks(); rev={k:defaultdict(list) for k in bs}
 for k,vals in bs.items():
  for x in vals: rev[k][letters(x)[::-1]].append(x)
 # Complete recombined clause: NP VP PP; reverse side uses same typed roles,
 # but every phrase is independently selected from the indexed bank.
 states=pruned=0; found=[]
 for np in bs.get('NP',()):
  for vp in bs.get('VP',()):
   for key,pps in rev.get('PP',{}).items():
    if states>=limit: break
    pp=pps[0]; states+=1
    l=letters(np+' '+vp); r=letters(pp)
    if consume(l,r) is None: pruned+=1; continue
    text=np+' '+vp+' '+pp
    if audit(text)['exact'] and len(letters(text))>38: found.append({'rendered':text,'audit':audit(text),'provenance':{'roles':['NP','VP','PP'],'reverse_index_queried':True,'recombined_phrase_units':True,'corpus_sentence_replayed':False}})
   if states>=limit: break
  if states>=limit: break
 return {'method':'typed-phrase-clause-transducer-20260919','bank_sizes':{k:len(v) for k,v in bs.items()},'states':states,'pruned':pruned,'exact_candidates':found,'candidate_count':len(found),'geometry':'single NP+VP+PP clause; not a complete mirrored two-sided grammar','status':'rejected incomplete geometry'}
if __name__=='__main__':
 o=run();(ROOT/'runs/typed-phrase-clause-transducer-20260919.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
