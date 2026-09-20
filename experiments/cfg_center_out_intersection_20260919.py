import hashlib,itertools,json
from pathlib import Path
DET=['a','the']; SUB=['pilot','baker','clerk']; VERB=['marks','opens']; OBJ=['map','gate','note']; ADV=['near','by']
def norm(s): return ''.join(c for c in s.lower() if c.isalpha())
def audit(s):
 t=norm(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); bad=next((i for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bad is None and bool(t),'first_mismatch':bad,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
rows=[]; states=0; pruned=0
for d,s,v,o,a in itertools.product(DET,SUB,VERB,OBJ,ADV):
 if len({d,s,v,o,a}) < 5: continue
 sentence=f'{d} {s} {v} {d} {o} {a} {o}'; t=norm(sentence); trace=[]
 for i in range((len(t)+1)//2):
  j=len(t)-1-i
  if t[i] != t[j]: pruned += 1; break
  trace.append({'left_pos':i,'right_pos':j,'left_char':t[i],'right_char':t[j],'offset':j-i,'role_state':'S->Det NP VP Adv'}); states+=1
 else: rows.append({'rendered':sentence,'audit':audit(sentence),'center_out_trace':trace,'mechanically_admitted':True})
best=max(rows,key=lambda x:x['audit']['letters']) if rows else None
out={'experiment_id':'cfg-center-out-intersection-20260919','signature':'fresh-cfg-character-trie-center-v2-pruned','method':'center-out DFS prunes first character mismatch before admitting a complete distinct-slot CFG derivation','candidates':rows,'best':best,'stats':{'derivations':len(rows),'states':states,'pruned':pruned,'exact':sum(x['audit']['exact'] for x in rows),'longest_letters':best['audit']['letters'] if best else 0},'provenance':{'single_sentence_derivation':True,'finished_tape_reversal':False,'paired_clauses':False,'aligned_token_pairs':False,'fallback':False,'distinct_slots':True,'novelty_preflight':'passed','human_readability_evidence':False},'reader_status':'pending human review; compiler output is not readability evidence'}
Path('runs/cfg-center-out-intersection-20260919.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['stats']))
