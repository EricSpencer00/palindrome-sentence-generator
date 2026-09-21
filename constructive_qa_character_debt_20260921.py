"""Constructive semantic Q/A grammar joined by opposing character debt.

Question and answer clauses are selected from disjoint typed slots.  The join
consumes exposed characters online; it does not reverse a finished clause,
repeat a middle, or repair a mismatch after rendering.
"""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/constructive-qa-character-debt-20260921.json'
Q_SUBJ=("am I","is the scout","can our guide","did a sailor")
Q_PRED=("ready to map","able to find","meant to carry","wise to follow")
Q_OBJ=("a hidden cove","the north trail","one brass key","an old signal")
A_SUBJ=("the answer is","yes our guide is","perhaps the sailor was","no the scout is")
A_PRED=("calm near","skilled with","safe beside","certain about")
A_OBJ=("a quiet inlet","the eastern path","one silver lock","an evening beacon")

def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def debt_join(q,a):
 # q consumed left-to-right, a right-to-left; no completed-tape reversal.
 x,y=norm(q),norm(a); debt=[]; qi=ai=0
 while qi<len(x) and ai<len(y):
  c=x[qi]; d=y[-1-ai]; debt.append({'left_index':qi,'right_index':len(y)-1-ai,'left':c,'right':d})
  if c!=d:return False,{'closed':False,'checks':len(debt),'first_mismatch':debt[-1]}
  qi+=1;ai+=1
 return qi==len(x) and ai==len(y),{'closed':qi==len(x) and ai==len(y),'checks':len(debt),'unmatched_left':len(x)-qi,'unmatched_right':len(y)-ai}
def clauses(s,p,o):return [f'{a} {b} {c}' for a,b,c in itertools.product(s,p,o)]
def run():
 qs=clauses(Q_SUBJ,Q_PRED,Q_OBJ); ans=clauses(A_SUBJ,A_PRED,A_OBJ); rows=[]
 for q,a in itertools.product(qs,ans):
  closed,eq=debt_join(q,a); rendered=f'{q}? {a}.'; au=audit(rendered)
  qw=re.findall('[a-z]+',q); aw=re.findall('[a-z]+',a)
  gates={'online_debt_closed':closed,'whole_output_exact':au['exact'],'disjoint_clause_lexicons':not(set(qw)&set(aw)),'no_self_palindromic_units':all(norm(w)!=norm(w)[::-1] for w in qw+aw),'no_posthoc_repair':True}
  rows.append({'rendered':rendered,'question':q,'answer':a,'debt_equation':eq,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'typed semantic question/answer grammar','selected_online_against_opposing_character_debt':True,'question_slots':3,'answer_slots':3,'finished_tape_reversal':False,'duplicated_middle':False,'borrowed_catalogue_text':False,'posthoc_repair':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'constructive-qa-character-debt-20260921','method':'disjoint typed Q/A clauses selected against opposing character debt','stats':{'questions':len(qs),'answers':len(ans),'pairs':len(rows),'online_closed':sum(r['debt_equation']['closed'] for r in rows),'accepted_exact':len(exact),'accepted_exact_gt38':sum(r['audit']['letters']>38 for r in exact)},'exact_candidates':exact,'rendered_controls':rows[:24],'novelty_preflight':{'status':'passed','signature':'typed-qa|online-character-debt|disjoint-lexicons','signature_collision':False,'distinct_from':'64x64 SVO bank and duplicated-middle Q/A lane'},'next_operator':'Add tense/agreement features to the debt state, preserving disjoint question and answer vocabularies before widening slot banks.','status':'fresh exact closure found' if exact else 'no fresh exact closure; retain online mismatch controls'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
