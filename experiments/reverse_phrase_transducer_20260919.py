"""Joint multiword phrase transducer; no corpus sentence replay or repair."""
from __future__ import annotations
import hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
from nltk.corpus import brown
ROOT=Path(__file__).resolve().parents[1]
def letters(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); return {'letters':len(t),'exact':bool(t) and t==t[::-1],'sha256':hashlib.sha256(t.encode()).hexdigest()}
def phrases():
 c=Counter();
 for sent in brown.tagged_sents():
  for i in range(len(sent)-1):
   ws=tuple(w.lower() for w,_ in sent[i:i+2]); ts=tuple(t for _,t in sent[i:i+2])
   if all(re.fullmatch('[a-z]+',w) for w in ws) and (ts[0].startswith(('AT','JJ','NN','VB')) or ts[0].startswith('IN')): c[ws]+=1
 return tuple(' '.join(x) for x,n in c.most_common(1600) if n>1)
def consume(a,b):
 n=min(len(a),len(b)); return (a[n:],b[:-n] if n else b) if a[:n]==b[::-1][:n] else None
def run(limit=120000):
 ps=phrases(); rev=defaultdict(list)
 for p in ps: rev[letters(p)[::-1]].append(p)
 states=pruned=0; found=[]
 for lp in ps:
  for key,rps in list(rev.items()):
   if states>=limit: break
   rp=rps[0]; states+=1
   if consume(letters(lp),letters(rp)) is None: pruned+=1; continue
   text=lp+' '+rp
   if audit(text)['exact'] and len(letters(text))>38: found.append({'rendered':text,'audit':audit(text),'provenance':{'phrase_units':2,'reverse_index_queried':True,'corpus_replay':False}})
  if states>=limit: break
 return {'method':'reverse-phrase-transducer-20260919','phrase_count':len(ps),'reverse_index_keys':len(rev),'states':states,'pruned':pruned,'exact_candidates':found,'candidate_count':len(found),'status':'no reader candidate' if not found else 'requires blinded reading'}
if __name__=='__main__':
 o=run(); (ROOT/'runs/reverse-phrase-transducer-20260919.json').write_text(json.dumps(o,indent=2)+'\n'); print(json.dumps(o,indent=2))
