"""Repair for the bilateral synchronizer: balanced adjuncts on both clauses."""
import json,hashlib,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'docs/experiment-novelty-registry.json'; OUT=ROOT/'runs/bilateral-slot-synchronizer-adjunct-repair-20260916.json'
ID='bilateral-slot-synchronizer-adjunct-repair-20260916'; SIG='bilateral-word-boundary-synchronizer|balanced-adjunct-repair|fresh-agreement-classes|complete-clause-pair|independent-two-pointer-audit'
def norm(s): return ''.join(c.lower() for c in s if 'a'<=c.lower()<='z')
def audit(s):
 t=norm(s); i,j=0,len(t)-1; first=None; n=0
 while i<j:
  n+=1
  if t[i]!=t[j] and first is None:first={'i':i,'j':j,'left':t[i],'right':t[j]}
  i+=1;j-=1
 return first is None, n, first
L=[('The baker packs bread','today'),('A quiet poet reads a map','outside'),('The guard watches the child','at dawn'),('Some young poets mark letters','with care')]
R=[('The sailor guides a child','today'),('A calm carver folds a ribbon','outside'),('The teacher opens the notebook','at dawn'),('Some kind clerks mark papers','with care')]
def run():
 d=json.load(open(REG)); prior={x['signature'] for x in d['entries'] if x['id']!=ID}
 if SIG in prior: raise RuntimeError('novelty collision')
 rows=[]
 for (l,la),(r,ra) in itertools.product(L,R):
  text=l+' '+la+'; '+r+' '+ra+'.'; ok,n,mm=audit(text)
  rows.append({'text':text,'letters':len(norm(text)),'exact':ok,'comparisons':n,'first_mismatch':mm,'provenance':'fresh balanced adjunct lexical classes; complete clauses'})
 ex=[x for x in rows if x['exact']]
 data={'experiment_id':ID,'signature':SIG,'branches':len(rows),'exact_count':len(ex),'rendered_candidates':ex,'rendered_probes':rows,'reader_eligible':[],'independent_audit':'two-pointer normalized tape comparison','base_artifact':'runs/bilateral-slot-synchronizer-20260916.json','provenance_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}; OUT.write_text(json.dumps(data,indent=2)+'\n'); print({'branches':len(rows),'exact':len(ex),'max_letters':max(x['letters'] for x in rows)})
if __name__=='__main__':run()
