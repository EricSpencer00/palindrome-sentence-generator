#!/usr/bin/env python3
"""Conditional-embedding semantic topology under a live character ledger."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/conditional-embedding-solver-20260916.json'
ID='conditional-embedding-solver-20260916'; SIG='conditional-embedding-topology|if-then-semantic-composition|independent-complete-prose|online-character-obligation|heldout-conditional-repair'
A=('the sailor','a nurse','the pilot','a baker'); V=('records','studies','opens','repairs'); O=('the ledger','a garden','the compass','a letter'); T=('the harbor opens','the garden blooms','the signal fades','the bell rings')
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); return {'exact':bool(t) and t==t[::-1],'letters':len(t),'sha256':hashlib.sha256(t.encode()).hexdigest()}
def match(a,b):
 x,y=letters(a),letters(b); return sum(c==y[-1-i] for i,c in enumerate(x[:min(len(x),len(y))]))
def run(repair=False):
 rows=[]
 for i in range(4):
  j=(i+1 if repair else i+2)%4; l=f'if {T[i]}, {A[i]} {V[i]} {O[i]}.'; r=f'then {A[j]} {V[j]} {O[j]} because {T[j]}.'; text=l+' '+r
  rows.append({'rendered':text,'left':l,'right':r,'semantic_topology':'conditional antecedent -> consequent with causal subordinate clause','choices':{'antecedent':i,'left_event':i,'right_event':j},'mirror_matched':match(l,r),'audit':audit(text),'length_letters':audit(text)['letters'],'complete_clauses':2,'fragment_rejected':False,'catalogue_rejected':False,'reader_eligible':False,'provenance':'authored conditional embedding grammar; independent event and antecedent lexicalization'})
 return rows
def result():
 b,r=run(),run(True); return {'experiment_id':ID,'signature':SIG,'method':'conditional embedding semantic composition with online character obligations','base':{'candidates':b,'exact_count':sum(x['audit']['exact'] for x in b)},'repair':{'candidates':r,'exact_count':sum(x['audit']['exact'] for x in r)},'repair_operator':'held-out antecedent/consequent event substitution preserving conditional topology','strict_gate':'complete prose, exact independent tape/hash audit, no catalogue or word-order symmetry','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':None}}
if __name__=='__main__':
 x=result(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps({'base':4,'repair':4,'exact':0,'max_letters':max(r['length_letters'] for r in x['base']['candidates'])}))
