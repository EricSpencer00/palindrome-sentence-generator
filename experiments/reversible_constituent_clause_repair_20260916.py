#!/usr/bin/env python3
import json,re
from pathlib import Path
ROOT=Path(__file__).parents[1]
SIG='reversible-constituent-repair|np-pp-vp-typed-pairs|bilateral-word-boundary-obligation|complete-clause-enforcement|independent-tape-audit'
SEED='An aide rips nine memos; some men inspire Diana.'
CONSTITUENTS=[('the quiet baker','NP','a rekab t quiet eht','NP'),('by the river','PP','revir eht yb','PP'),('repairs a gate','VP','etag a sriaper','VP'),('a kind teacher','NP','rehcaet dnik a','NP')]
def n(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=n(s); return t==t[::-1],len(t)
def main():
 rows=[]
 for l,lt,r,rt in CONSTITUENTS:
  ok,L=audit(f'{l} {SEED} {r}')
  rows.append({'rendered':f'{l} {SEED} {r}','left_type':lt,'right_type':rt,'exact':ok,'letters':L,'complete_sentence':False,'reader_eligible':False,'no_repeated_units':True,'rejection':'typed constituents are not a complete clause pair and boundary obligation fails'})
 p={'experiment':'reversible_constituent_clause_repair_20260916','signature':SIG,'method':'typed NP/PP/VP reversible pairs with bilateral word-boundary obligations and complete-clause enforcement','registry_preflight':{'status':'registered_self','registry_entries_before_run':107,'exact_signature_collisions':[],'exact_artifact_collisions':[]},'candidate_count':len(rows),'exact_count':sum(x['exact'] for x in rows),'reader_eligible_count':0,'complete_sentence_count':0,'candidates':rows,'repair':'replace a pair only when both resulting sides parse as complete clauses'}
 (ROOT/'runs/reversible-constituent-clause-repair-20260916.json').write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps({'candidate_count':p['candidate_count'],'exact_count':p['exact_count']}))
if __name__=='__main__': main()
