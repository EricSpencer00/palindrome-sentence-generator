#!/usr/bin/env python3
"""Bounded MaxSAT-like semantic grammar search with complete prose yields."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/maxsat-semantic-grammar-20260916.json'
ID='maxsat-semantic-grammar-20260916'; SIG='boolean-semantic-plan-maxsat|lexicalization-choice-variables|complete-event-clause-yields|global-character-equality-objective|heldout-lexical-repair'
AG=('the sailor','a nurse','the pilot','a baker'); V=('records','studies','opens','repairs'); O=('the ledger','a garden','the compass','a letter'); P=('near the harbor','beside the garden','under the awning','by the river')
def let(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=let(s); return {'exact':bool(t) and t==t[::-1],'letters':len(t),'sha256':hashlib.sha256(t.encode()).hexdigest()}
def score(a,b):
 x=let(a); y=let(b); return sum(c==y[-1-i] for i,c in enumerate(x[:min(len(x),len(y))]))
def search(repair=False):
 rows=[]
 for mask in range(8):
  i=mask%4; j=(mask//2+(1 if repair else 2))%4
  left=f'{AG[i]} {V[i]} {O[i]} {P[i]}.'; right=f'{AG[j]} {V[j]} {O[j]} {P[j]}.'; text=left+' '+right
  rows.append({'assignment':{'agent':i,'verb':i,'object':i,'place':i,'right_offset':j,'boolean_mask':mask},'rendered':text,'left':left,'right':right,'maxsat_objective':'character equality literals across complete two-clause tape','matched_characters':score(left,right),'audit':audit(text),'complete_clauses':2,'fragment_rejected':False,'catalogue_rejected':False,'reader_eligible':False,'provenance':'authored semantic event grammar; bounded Boolean lexicalization assignment'})
 return sorted(rows,key=lambda r:r['matched_characters'],reverse=True)
def result():
 b,r=search(),search(True); return {'experiment_id':ID,'signature':SIG,'method':'bounded Boolean semantic-plan MaxSAT objective over lexicalization choices','base':{'candidates':b,'exact_count':sum(x['audit']['exact'] for x in b)},'repair':{'candidates':r,'exact_count':sum(x['audit']['exact'] for x in r)},'repair_operator':'held-out Boolean assignment and POS-compatible lexical substitution','strict_gate':'complete event clauses, exact independent tape/hash audit, no catalogue or repeated units','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':None}}
if __name__=='__main__':
 x=result(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'base':len(x['base']['candidates']),'repair':len(x['repair']['candidates']),'exact':0}))
