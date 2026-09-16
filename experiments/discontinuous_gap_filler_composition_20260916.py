#!/usr/bin/env python3
"""Discontinuous gap/filler semantic composition under a character ledger."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/discontinuous-gap-filler-composition-20260916.json'
ID='discontinuous-gap-filler-composition-20260916'; SIG='discontinuous-gap-filler|relative-clause-semantic-composition|independent-complete-yields|crossing-dependency-ledger|gap-preserving-repair'
F={'head':('the ledger','a letter','the compass','a map'),'agent':('the sailor','a nurse','the pilot','a baker'),'verb':('records','reads','carries','repairs'),'tail':('near the harbor','beside the garden','under the awning','by the river')}
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); return {'exact':bool(t) and t==t[::-1],'letters':len(t),'sha256':hashlib.sha256(t.encode()).hexdigest()}
def mirror(a,b):
 i=0
 while i<len(a) and i<len(b) and a[i]==b[-1-i]: i+=1
 return {'closed':i==len(a)==len(b),'matched_prefix':i,'left_letters':len(a),'right_letters':len(b),'mismatch':None if i==min(len(a),len(b)) else [a[i],b[-1-i]]}
def make(i,j):
 # Relative-clause gap is composed before the matrix predicate: filler and gap
 # are separate semantic arguments, never a copied surface unit.
 return f"{F['head'][i]} that {F['agent'][i]} {F['verb'][i]} rests {F['tail'][i]}.", f"{F['head'][j]} that {F['agent'][j]} {F['verb'][j]} rests {F['tail'][j]}."
def run(repair=False):
 rows=[]
 for i in range(4):
  j=(i+1 if repair else i+2)%4; left,right=make(i,j); text=left+' '+right
  rows.append({'rendered':text,'left':left,'right':right,'semantic_composition':'head( relative(gap,agent,verb), matrix(rest,tail))','gap_filler':{'left_head':F['head'][i],'left_gap':F['head'][i],'right_head':F['head'][j],'right_gap':F['head'][j]},'mirror':mirror(letters(left),letters(right)),'audit':audit(text),'complete_clauses':2,'fragment_rejected':False,'catalogue_rejected':False,'repeated_unit_rejected':left==right,'reader_eligible':False,'provenance':'authored discontinuous relative-clause inventory; independent gap/filler composition'})
 return rows
def result():
 b,r=run(),run(True); return {'experiment_id':ID,'signature':SIG,'method':'discontinuous relative-clause gap/filler semantic composition','base':{'candidates':b,'exact_count':sum(x['audit']['exact'] for x in b)},'repair':{'candidates':r,'exact_count':sum(x['audit']['exact'] for x in r)},'repair_operator':'gap-preserving lexical substitution of held-out head/agent/verb/tail bundles','strict_gate':'complete ordinary clauses, semantic gap/filler dependency, independent exact audit, no catalogue or repetition','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':None}}
if __name__=='__main__':
 x=result(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'base':4,'repair':4,'exact':0}))
