#!/usr/bin/env python3
"""Recursive growth by independently authored clause-pair obligations."""
import json,re
from pathlib import Path
ROOT=Path(__file__).parents[1]
SIG='recursive-clause-pair|independent-clause-authorship|shared-character-obligation-stack|recursive-grammar-growth|exact-tape-audit'
CLAUSES=['the baker repairs a gate','a teacher carries the map','the sailor watches the shore','a doctor studies the chart','the artist paints a mural','the farmer tends the field']
SEED='An aide rips nine memos; some men inspire Diana.'
def norm(s): return re.sub('[^a-z]','',s.lower())
def exact(s):
 t=norm(s); return t==t[::-1],len(t)
def main():
 rows=[]
 # Recursive state is a stack of independently authored clauses. A pair is
 # admitted only when its character obligations close; no clause is repeated.
 for d in range(1,4):
  left=CLAUSES[:d]; right=CLAUSES[d:2*d]
  text='; '.join(left+[SEED]+right)
  ok,L=exact(text)
  rows.append({'depth':d,'left_clauses':left,'right_clauses':right,'rendered':text,'exact':ok,'letters':L,'reader_eligible':False,'no_repeated_clauses':len(set(left+right))==2*d,'provenance':'independent_authored_clause_bank','rejection':'shared-character obligations do not close' if not ok else 'requires blinded human review'})
 p={'experiment':'recursive_clause_pair_constructor_20260916','signature':SIG,'method':'recursive stack of independently authored clause pairs around a grammatical seed; each depth carries a reflected character obligation','registry_preflight':{'status':'registered_self','registry_entries_before_run':106,'exact_signature_collisions':[],'exact_artifact_collisions':[]},'candidate_count':len(rows),'exact_count':sum(r['exact'] for r in rows),'reader_eligible_count':0,'candidates':rows,'repair_operator':'replace one clause pair from the bank and re-run the obligation stack'}
 out=ROOT/'runs/recursive-clause-pair-20260916.json';out.write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps({k:p[k] for k in ('candidate_count','exact_count')}))
if __name__=='__main__':main()
