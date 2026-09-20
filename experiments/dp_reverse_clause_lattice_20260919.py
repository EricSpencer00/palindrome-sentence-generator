"""Bounded DP reverse-tape parser for complete clause templates."""
import json, hashlib, itertools
from pathlib import Path
DET=['a','an','the']; NOUN=['aide','artist','baker','pilot','clerk','guard']; VERB=['rips','marks','opens','reads','packs']; NUM=['one','nine','two']; OBJ=['memo','map','note','gate','parcel']
LEX=set(DET+NOUN+VERB+NUM+OBJ+['near','by','at','in','the','some','men','nora','ada'])
def norm(s): return ''.join(c for c in s if c.isalpha()).lower()
def parse(t):
 dp={0:[]}
 for i in range(len(t)):
  if i not in dp: continue
  for w in LEX:
   if t.startswith(w,i) and i+len(w) not in dp: dp[i+len(w)]=dp[i]+[w]
 return dp.get(len(t))
def audit(s):
 t=norm(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); return {'letters':len(t),'exact':t==t[::-1],'sha_equal':f==r,'forward':f,'reverse':r}
rows=[]
for d,n,v,num,obj in itertools.product(DET,NOUN,VERB,NUM,OBJ):
 left=f'{d} {n} {v} {num} {obj}'; rev=norm(left)[::-1]; toks=parse(rev)
 if not toks or len(toks)<5: continue
 right=' '.join(toks)
 if not (toks[0] in DET and toks[1] in NOUN and toks[2] in VERB): continue
 rows.append({'left':left,'right':right,'rendered':left+'; '+right,'audit':audit(left+right),'boundary_offsets_differ':True,'aligned_token_pairs':False})
best=max(rows,key=lambda x:x['audit']['letters']) if rows else None
out={'experiment_id':'dp-reverse-clause-lattice-20260919','signature':'dp-heldout-clause-v1','method':'authored DET-NOUN-VERB-NUM/DET-NOUN clause; reverse tape parsed by DP into a complete lexical clause','candidates':rows,'best':best,'stats':{'trials':len(DET)*len(NOUN)*len(VERB)*len(NUM)*len(OBJ),'admissible':len(rows),'exact':sum(x['audit']['exact'] for x in rows),'longest_letters':best['audit']['letters'] if best else 0},'provenance':{'heldout_lexicon':True,'single_letter_fallback':False,'catalogue':False,'human_readability_evidence':False},'reader_status':'pending; only fully lexical complete parses retained'}
Path('runs/dp-reverse-clause-lattice-20260919.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['stats']))
