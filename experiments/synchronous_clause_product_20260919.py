import hashlib,json,re,itertools
from pathlib import Path
R=Path(__file__).resolve().parents[1]; W=re.compile('[a-z]+')
S=("a careful pilot","the patient keeper","a quiet singer","the gardener")
V=("maps","guards","carries","marks")
O=("rivers","old charts","small parcels","western roads")
A=("at dawn","near home","with care","by noon")
def n(x):return ''.join(W.findall(x.lower()))
def au(x):
 t=n(x);i=0;j=len(t)-1
 while i<j and t[i]==t[j]:i+=1;j-=1
 return {'exact':i>=j,'first_mismatch':None if i>=j else [i,j,t[i],t[j]],'letters':len(t),'sha_forward':hashlib.sha256(t.encode()).hexdigest(),'sha_reverse':hashlib.sha256(t[::-1].encode()).hexdigest(),'two_pointer_exact':i>=j}
L=[f'{s} {v} {o} {a}' for s,v,o,a in itertools.product(S,V,O,A)]
rows=[]; states=0
for l,r in itertools.product(L,L):
 if l==r: continue
 # synchronous lexical expansion gate: consume each chosen word pair's
 # available outer orbit before allowing the next phrase slot.
 ok=True
 for x,y in zip(l.split(),reversed(r.split())):
  states+=1
  if x[0]!=y[-1]: ok=False; break
 if ok:
  z=f'{l}; a; {r}'; q=au(z)
  if q['letters']>38: rows.append({'text':z,'audit':q,'provenance':{'synchronous':True,'catalogue':False,'finished_reversal':False,'repeated':False}})
out={'experiment':'synchronous-clause-product-20260919','grammar_sizes':{'clauses':len(L)},'states':states,'candidates':rows,'exact':sum(x['audit']['exact'] for x in rows),'next_discriminator':'character-indexed lexical expansion trie with held-out subject/object phrases'}
(R/'runs').mkdir(exist_ok=True);(R/'runs/synchronous-clause-product-20260919.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'clauses':len(L),'states':states,'candidates':len(rows),'exact':out['exact']}))
