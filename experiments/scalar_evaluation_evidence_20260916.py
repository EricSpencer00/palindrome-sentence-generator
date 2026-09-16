#!/usr/bin/env python3
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/scalar-evaluation-evidence-20260916.json'
ID='scalar-evaluation-evidence-20260916'; SIG='scalar-evaluation-predicate|independent-evidence-clause|semantic-judgment-composition|online-character-obligation|heldout-evaluation-repair'
A=('the sailor','a nurse','the pilot','a baker'); E=('the harbor is calm','the garden is green','the signal is clear','the bell is loud')
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); return {'exact':bool(t) and t==t[::-1],'letters':len(t),'sha256':hashlib.sha256(t.encode()).hexdigest()}
def run(repair=False):
 rows=[]
 for i in range(4):
  j=(i+1 if repair else i+2)%4; left=f'{A[i]} finds the scene sound: {E[i]}.'; right=f'{A[j]} judges the report fair: {E[j]}.'; text=left+' '+right
  rows.append({'rendered':text,'left':left,'right':right,'semantic_topology':'event -> scalar evaluation predicate -> evidence proposition','choices':{'evaluator':i,'evidence':i,'right_evaluator':j},'audit':audit(text),'length_letters':audit(text)['letters'],'complete_clauses':2,'fragment_rejected':False,'catalogue_rejected':False,'reader_eligible':False,'provenance':'authored evaluation/evidence grammar; independent lexicalization'})
 return rows
def result():
 b,r=run(),run(True); return {'experiment_id':ID,'signature':SIG,'method':'scalar evaluation predicate with independent evidence clause','base':{'candidates':b,'exact_count':0},'repair':{'candidates':r,'exact_count':0},'repair_operator':'held-out evaluator and evidence substitution preserving judgment topology','strict_gate':'complete prose, independent exact/hash audit, no catalogue or word-order symmetry','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':None}}
if __name__=='__main__':
 x=result(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'base':4,'repair':4,'exact':0,'max_letters':max(r['length_letters'] for r in x['base']['candidates'])}))
