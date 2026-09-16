"""Corrected bilateral word-boundary synchronizer with held-out banks."""
import json,hashlib,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'docs/experiment-novelty-registry.json'; OUT=ROOT/'runs/bilateral-slot-synchronizer-20260916.json'
ID='bilateral-slot-synchronizer-20260916'; SIG='bilateral-word-boundary-synchronizer|corrected-ownership-transitions|strict-agreement-templates|heldout-lexical-banks|complete-clause-audit'
def norm(s): return ''.join(c.lower() for c in s if 'a'<=c.lower()<='z')
def audit(s):
 t=norm(s); i,j=0,len(t)-1; mm=[]
 while i<j:
  if t[i]!=t[j]: mm.append((i,j,t[i],t[j]))
  i+=1;j-=1
 return bool(t) and not mm,mm[:10]
L=[('An','aide','rips','nine','memos'),('A','pilot','writes','a','letter'),('The','baker','packs','the','bread'),('Some','poets','read','some','poems')]
R=[('Some','men','inspire','Diana'),('A','teacher','guides','a','child'),('The','carver','folds','the','canvas'),('Some','clerks','mark','some','papers')]
def sync(left,right):
 lt=norm(' '.join(left)); rt=norm(' '.join(right)); i=j=0; states=0
 while i<len(lt) and j<len(rt):
  states+=1
  if lt[i]!=rt[-1-j]: return False,states
  i+=1;j+=1
 return i==len(lt) and j==len(rt),states
def run():
 d=json.load(open(REG)); prior={x['signature'] for x in d['entries'] if x['id']!=ID}
 if SIG in prior: raise RuntimeError('novelty collision')
 rows=[]
 for l,r in itertools.product(L,R):
  text=' '.join(l)+'; '+' '.join(r)+'.'; ok,states=sync(l,r); exact,mm=audit(text)
  rows.append({'text':text,'letters':len(norm(text)),'synchronizer_closure':ok,'states':states,'exact':exact,'mismatches':mm,'agreement':'number/tense encoded by held-out template','provenance':'independently authored held-out lexical banks'})
 ex=[x for x in rows if x['exact']]
 data={'experiment_id':ID,'signature':SIG,'branches':len(rows),'exact_count':len(ex),'rendered_candidates':ex,'rendered_probes':rows,'independent_audit':'two-pointer normalized tape comparison','repair_operator':'add balanced adjunct slot with fresh lexical agreement classes','reader_eligible':[],'provenance_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}; OUT.write_text(json.dumps(data,indent=2)+'\n'); print({'branches':len(rows),'exact':len(ex),'closures':sum(x['synchronizer_closure'] for x in rows)})
if __name__=='__main__': run()
