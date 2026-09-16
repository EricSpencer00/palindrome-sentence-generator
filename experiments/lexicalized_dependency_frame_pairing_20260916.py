"""Lexicalized dependency-frame pairing with semantic state retained."""
import json,hashlib,itertools,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'docs/experiment-novelty-registry.json'; OUT=ROOT/'runs/lexicalized-dependency-frame-pairing-20260916.json'
ID='lexicalized-dependency-frame-pairing-20260916'; SIG='lexicalized-dependency-frame-pairing|role-valency-state|independent-agent-patient-realization|cross-frame-character-join|semantic-repair-operator'
def norm(s): return ''.join(c.lower() for c in s if 'a'<=c.lower()<='z')
def audit(s):
 t=norm(s); i,j=0,len(norm(s))-1; mm=[]
 while i<j:
  if t[i]!=t[j]: mm.append({'i':i,'j':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 return bool(t) and not mm,mm[:10]
LEFT=[('baker','pack','bread','transitive'),('poet','read','map','transitive'),('guard','watch','child','transitive'),('sailor','wait',None,'intransitive'),('nurse','help','patient','transitive')]
RIGHT=[('carver','fold','canvas','transitive'),('teacher','guide','child','transitive'),('keeper','hold','letter','transitive'),('dancer','smile',None,'intransitive'),('clerk','mark','paper','transitive')]
def run():
 d=json.load(open(REG)); prior={x['signature'] for x in d['entries'] if x['id']!=ID}
 if SIG in prior: raise RuntimeError('novelty collision')
 rows=[]
 for (la,lv,lo,lr),(ra,rv,ro,rr) in itertools.product(LEFT,RIGHT):
  if lr!=rr: continue
  left=f"The {la} {lv}"+(f" the {lo}" if lo else '')
  right=f"The {ra} {rv}"+(f" the {ro}" if ro else '')
  text=left+'; '+right+'.'; ok,mm=audit(text)
  rows.append({'text':text,'letters':len(norm(text)),'exact':ok,'mismatches':mm,'semantic_state':{'left_role':lr,'right_role':rr,'left_agent':la,'right_agent':ra},'provenance':'independent lexicalized dependency frames'})
 exact=[r for r in rows if r['exact']]
 data={'experiment_id':ID,'signature':SIG,'frames_checked':len(rows),'exact_count':len(exact),'rendered_probes':rows,'rendered_candidates':exact,'repair_operator':'add causative/result-state frame with independently typed patient roles','reader_eligible':[],'independent_audit':'two-pointer normalized character comparison','provenance_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}; OUT.write_text(json.dumps(data,indent=2)+'\n'); print({'frames':len(rows),'exact':len(exact)})
if __name__=='__main__': run()
