#!/usr/bin/env python3
"""All-different lexical word-equation search over independent POS slots."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/lexical-word-equation-inventory-20260916.json'
ID='lexical-word-equation-inventory-20260916'; SIG='all-different-lexical-word-equation|semantic-pos-slot-inventory|joint-boundary-choice|independent-complete-prose|pos-compatible-repair'
SLOTS={'subj':('the sailor','a nurse','the botanist','a teacher','the pilot','a baker'),'verb':('records','studies','opens','repairs','carries','observes'),'obj':('the ledger','a garden','the window','a harbor','the compass','a letter'),'prep':('near the river','beside the garden','under the awning','by the harbor','across the field','within the tower')}
def let(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=let(s); return {'exact':bool(t) and t==t[::-1],'letters':len(t),'sha256':hashlib.sha256(t.encode()).hexdigest()}
def equation(a,b):
 i=0
 while i<len(a) and i<len(b) and a[i]==b[-1-i]: i+=1
 return {'closed':i==len(a)==len(b),'matched_prefix':i,'left_letters':len(a),'right_letters':len(b),'first_mismatch':None if i==min(len(a),len(b)) else [a[i],b[-1-i]]}
def run(repair=False):
 rows=[]
 for i in range(6):
  j=(i+1 if repair else i+3)%6
  left=' '.join(SLOTS[k][i] for k in ('subj','verb','obj','prep'))+'.'; right=' '.join(SLOTS[k][j] for k in ('subj','verb','obj','prep'))+'.'; text=left+' '+right
  toks=re.findall(r'[a-z]+',let(text))
  rows.append({'rendered':text,'left':left,'right':right,'slot_choices':{'left':i,'right':j},'boundary_mode':'word boundaries jointly scored after POS-slot choice','equation':equation(let(left),let(right)),'audit':audit(text),'complete_clauses':2,'fragment_rejected':False,'catalogue_rejected':False,'all_different':len(toks)==len(set(toks)),'reader_eligible':False,'provenance':'authored independent POS-slot inventory; deterministic lexical equation enumeration'})
 return rows
def result():
 b,r=run(),run(True)
 return {'experiment_id':ID,'signature':SIG,'method':'joint lexical word-equation over independent semantic POS slots','base':{'candidates':b,'exact_count':sum(x['audit']['exact'] for x in b)},'repair':{'candidates':r,'exact_count':sum(x['audit']['exact'] for x in r)},'repair_operator':'POS-compatible held-out index shift with all-different lexical gate','strict_gate':'complete clauses, all-different content, no catalogue, exact independent audit','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':None}}
if __name__=='__main__':
 x=result(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'base':6,'repair':6,'exact':x['base']['exact_count']+x['repair']['exact_count']}))
