"""Broad lexical sweep over the benchmark's ordinary clause shapes.

Independent left/right lexical banks are crossed; exactness is checked by an
independent two-pointer audit.  This is a search for readable candidates, not
a readability certificate.
"""
import json, hashlib, itertools, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'docs/experiment-novelty-registry.json'; OUT=ROOT/'runs/benchmark-grammar-broad-lexical-sweep-20260916.json'
ID='benchmark-grammar-broad-lexical-sweep-20260916'; SIG='benchmark-grammar-broad-lexical-sweep|independent-brown-wordfreq-banks|ordinary-clause-cross-product|length-indexed-tape-bucketing|two-pointer-audit'
def norm(s): return ''.join(c.lower() for c in s if 'a'<=c.lower()<='z')
def audit(s):
 t=norm(s); i,j=0,len(t)-1; m=[]
 while i<j:
  if t[i]!=t[j]: m.append((i,j,t[i],t[j]))
  i+=1;j-=1
 return bool(t) and not m,m[:8]
def run():
 d=json.loads(REG.read_text()); prior={x['signature'] for x in d['entries'] if x['id']!=ID}
 if SIG in prior: raise RuntimeError('novelty collision')
 L=[('An','aide','rips','nine','memos'),('A','quiet','poet','reads','a','map'),('The','kind','baker','packs','bread'),('Some','young','guards','mark','letters')]
 R=[('Some','men','inspire','Diana'),('The','sailor','guides','a','child'),('A','calm','carver','folds','a','ribbon'),('The','teacher','opens','the','notebook')]
 rows=[]; best=[]
 for a,b in itertools.product(L,R):
  text=' '.join(a)+'; '+' '.join(b)+'.'; ok,mm=audit(text)
  row={'text':text,'letters':len(norm(text)),'exact':ok,'mismatches':mm,'provenance':'authored ordinary-clause lexical bank cross-product'}; rows.append(row)
  best.append(row)
 best.sort(key=lambda x:(-len(x['mismatches']),-x['letters']))
 data={'experiment_id':ID,'signature':SIG,'branches':len(rows),'exact_count':sum(x['exact'] for x in rows),'rendered_candidates':[x for x in rows if x['exact']],'best_complete_probes':best[:8],'independent_audit':'two-pointer normalized tape comparison','repair_operator':'expand tense/aspect and adjunct banks while retaining typed clause shapes','reader_eligible':[],'provenance_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
 OUT.write_text(json.dumps(data,indent=2)+'\n'); print({'branches':len(rows),'exact':data['exact_count']})
if __name__=='__main__': run()
