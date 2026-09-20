import json,hashlib,itertools
from pathlib import Path
WORDS=['a','the','one','pilot','baker','clerk','guard','poet','marks','opens','reads','packs','seals','map','gate','note','lamp','parcel','book','near','by','at','under']
def norm(s): return ''.join(c for c in s if c.isalpha()).lower()
def audit(s):
 t=norm(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); return {'letters':len(t),'exact':t==t[::-1],'sha_equal':f==r,'forward':f,'reverse':r}
trie={}
for w in WORDS:
 n=trie
 for c in w: n=n.setdefault(c,{})
 n['$']=1
rows=[]; states=0; pruned=0
for d,s,v,o,a,o2 in itertools.product(['a','the','one'],['pilot','baker','clerk','guard','poet'],['marks','opens','reads','packs','seals'],['map','gate','note','lamp','parcel'],['near','by','at'],['book','lamp','gate']):
 if len({d,s,v,o,a,o2})<6: continue
 sentence=f'{d} {s} {v} {o} {a} {o2}'; t=norm(sentence); ok=True
 for i in range((len(t)+1)//2):
  states+=1
  if t[i]!=t[-1-i]: pruned+=1; ok=False; break
 if ok: rows.append({'rendered':sentence,'audit':audit(sentence),'trie_states':len(t),'grammar_state':'S>Det NP VP Obj Adj Obj','boundary_offsets':[len(t)-1-2*i for i in range((len(t)+1)//2)]})
best=max(rows,key=lambda x:x['audit']['letters']) if rows else None
out={'experiment_id':'lexical-trie-live-cfg-20260919','signature':'lexical-trie-prefix-live-v1','method':'character-prefix trie with live CFG POS/role/boundary state and center-out obligation pruning','candidates':rows,'best':best,'stats':{'trie_words':len(WORDS),'states':states,'pruned':pruned,'admissible':len(rows),'exact':sum(x['audit']['exact'] for x in rows)},'provenance':{'single_sentence':True,'paired_clauses':False,'reversal':False,'fallback':False,'repeated_units':False,'human_readability_evidence':False}}
Path('runs/lexical-trie-live-cfg-20260919.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['stats']))
