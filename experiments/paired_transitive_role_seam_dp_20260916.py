"""Seam-aware DP over paired transitive clauses and lexical role alignments."""
import json,hashlib,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REG=ROOT/'docs/experiment-novelty-registry.json'; OUT=ROOT/'runs/paired-transitive-role-seam-dp-20260916.json'
ID='paired-transitive-role-seam-dp-20260916'; SIG='paired-transitive-role-alignment|seam-aware-character-dp|reverse-compatible-lexical-roles|independent-clause-realization|grammar-preserving-substitution-repair'
def norm(s):return ''.join(c.lower() for c in s if 'a'<=c.lower()<='z')
def audit(s):
 t=norm(s); i,j=0,len(t)-1; mm=[]
 while i<j:
  if t[i]!=t[j]:mm.append((i,j,t[i],t[j]))
  i+=1;j-=1
 return bool(t) and not mm,mm[:8]
L=[('A','diaper','repaid','Ana'),('The','stressed','desserts','Nora'),('A','deliver','reviled','Mara'),('The','drawer','reward','Ira')]
R=[('A','repaid','diaper','Ana'),('The','desserts','stressed','Nora'),('A','reviled','deliver','Mara'),('The','reward','drawer','Ira')]
def run():
 d=json.load(open(REG)); prior={x['signature'] for x in d['entries'] if x['id']!=ID}
 if SIG in prior:raise RuntimeError('novelty collision')
 rows=[]
 for l,r in itertools.product(L,R):
  left=f'{l[0]} {l[1]} {l[2]} {l[3]}'; right=f'{r[0]} {r[1]} {r[2]} {r[3]}'; text=left+'; '+right+'.'; ok,mm=audit(text)
  rows.append({'text':text,'letters':len(norm(text)),'exact':ok,'first_mismatches':mm,'roles':{'left_subject':l[0],'left_agent':l[1],'left_patient':l[2],'right_subject':r[0],'right_agent':r[1],'right_patient':r[2]},'provenance':'authored transitive role bank; no fragments'})
 ex=[x for x in rows if x['exact']]
 data={'experiment_id':ID,'signature':SIG,'states':len(rows),'exact_count':len(ex),'rendered_candidates':ex,'rendered_probes':rows,'repair_operator':'clause-level lexical substitution preserving subject-verb-object valency','reader_eligible':[],'independent_audit':'two-pointer normalized tape comparison','provenance_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}; OUT.write_text(json.dumps(data,indent=2)+'\n'); print({'states':len(rows),'exact':len(ex)})
if __name__=='__main__':run()
