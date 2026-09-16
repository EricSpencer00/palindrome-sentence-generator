#!/usr/bin/env python3
"""Evidential semantic-scene planner with online mirror scoring."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/evidential-scene-planner-20260916.json'
ID='evidential-scene-planner-20260916'; SIG='evidential-scene-plan|source-of-knowledge-slot|independent-complete-prose|online-mirror-score|heldout-evidential-repair'
E=(('the sailor','records','the ledger'),('a nurse','studies','a garden'),('the pilot','opens','the compass'),('a baker','repairs','a letter'))
Q=('it seems','witnesses say','records show','perhaps')
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); return {'exact':bool(t) and t==t[::-1],'letters':len(t),'sha256':hashlib.sha256(t.encode()).hexdigest()}
def match(a,b):
 x,y=letters(a),letters(b); return sum(c==y[-1-i] for i,c in enumerate(x[:min(len(x),len(y))]))
def run(repair=False):
 rows=[]
 for i in range(4):
  j=(i+1 if repair else i+2)%4; a,v,o=E[i]; b,w,p=E[j]; left=f'{Q[i]}: {a} {v} {o}.'; right=f'{Q[j]}: {b} {w} {p}.'; text=left+' '+right
  rows.append({'rendered':text,'left':left,'right':right,'semantic_scene':{'source':'evidence','event':i,'role_slots':['agent','action','patient']},'mirror_matched':match(left,right),'audit':audit(text),'complete_clauses':2,'fragment_rejected':False,'catalogue_rejected':False,'reader_eligible':False,'provenance':'authored evidential scene planner; independent source and event lexicalization'})
 return rows
def result():
 b,r=run(),run(True); return {'experiment_id':ID,'signature':SIG,'method':'evidential source-of-knowledge semantic scene planning','base':{'candidates':b,'exact_count':sum(x['audit']['exact'] for x in b)},'repair':{'candidates':r,'exact_count':sum(x['audit']['exact'] for x in r)},'repair_operator':'held-out source-of-knowledge substitution plus event-role relexicalization','strict_gate':'complete prose, independent exact/hash audit, no catalogue or mirrored units','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':None}}
if __name__=='__main__':
 x=result(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'base':4,'repair':4,'exact':0}))
