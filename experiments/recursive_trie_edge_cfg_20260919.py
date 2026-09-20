import json,hashlib
from pathlib import Path
LEX=['a','the','pilot','baker','marks','opens','map','gate','near','by']; trie={}
for w in LEX:
 n=trie
 for c in w: n=n.setdefault(c,{})
 n['$']=1
trans=0; pruned=0; rendered=[]
def words(node,p=''):
 global trans
 for c,ch in node.items():
  if c=='$': yield p
  else:
   trans+=1; yield from words(ch,p+c)
bank=list(words(trie))
for a in bank:
 for b in bank:
  if a==b: continue
  prefix=a+' '+b
  if prefix[0]!=prefix[-1]: pruned+=1; continue
  rendered.append(prefix)
def audit(s):
 t=''.join(c for c in s if c.isalpha()); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); return {'letters':len(t),'exact':t==t[::-1],'sha_equal':f==r,'forward':f,'reverse':r}
rows=[{'rendered':s,'audit':audit(s)} for s in rendered]
out={'experiment_id':'recursive-trie-edge-cfg-20260919','signature':'recursive-trie-edge-live-v1','method':'recursive trie-edge traversal with distinct lexical choices and immediate outer-character pruning','candidates':rows,'stats':{'trie_words':len(bank),'transitions':trans,'pruned_prefixes':pruned,'rendered':len(rows),'exact':sum(x['audit']['exact'] for x in rows)},'provenance':{'real_trie_traversal':True,'finished_tape_reversal':False,'fallback':False,'human_readability_evidence':False}}
Path('runs/recursive-trie-edge-cfg-20260919.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['stats']))
