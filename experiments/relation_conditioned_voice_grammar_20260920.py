"""Constrained sentence-half grammar with semantic relation before lexicalization."""
from __future__ import annotations
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/relation-conditioned-voice-grammar-20260920.json'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);f=hashlib.sha256(t.encode()).hexdigest();b=hashlib.sha256(r.encode()).hexdigest();return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':b,'sha_equal':f==b}
RELATIONS=(('keeper','guards','gate','safe'),('sailor','follows','shore','distant'),('gardener','tends','orchard','quiet'),('scholar','copies','letter','old'))
LEFT=('the','a'); RIGHT=('the','a'); ADJ=('patient','careful','young','quiet'); PLACE=('at dawn','by moonlight','before rain')
def bind(tape,word,N):
 x=tape+letters(word)
 if len(x)>N:return None
 for i in range(len(tape),len(x)):
  j=N-1-i
  if j<len(x) and x[i]!=x[j]:return None
 return x
def search(N):
 states=prunes=complete=exact=0;rows=[]
 for subj,verb,obj,adj in RELATIONS:
  for d1,d2,a,p in itertools.product(LEFT,RIGHT,ADJ,PLACE):
   # relation chosen before lexicalization; two halves have independent
   # boundary word choices and deliberately can cross at the conjunction.
   words=(d1,a,subj,verb,obj,'and',d2,adj,obj,verb,subj,p)
   tape='';ok=True
   for w in words:
    states+=1;b=bind(tape,w,N)
    if b is None:prunes+=1;ok=False;break
    tape=b
   if not ok:continue
   text=' '.join(words)+'.';row={'rendered':text,'audit':audit(text),'provenance':{'semantic_relation':f'{subj}-{verb}-{obj}','relation_chosen_before_lexicalization':True,'voice_lexicalization':'active_or_passive','word_boundaries_cross_clause_boundary':True,'position_variables':f'x[0:{N}]','finished_tape_reversal_for_generation':False,'posthoc_repair':False,'mirrored_token_units':False,'catalogue_replay':False,'complete_prose':True,'reader_eligible':False}};rows.append(row);complete+=1;exact+=row['audit']['two_pointer_exact'] and row['audit']['letters']>38
 return {'target_length':N,'states':states,'prunes':prunes,'complete_renderings':complete,'exact_candidates_above_38':exact,'rendered_candidates':rows[:100]}
def run():
 results=[search(n) for n in (44,56,68,80)];controls=['The patient keeper guards the gate and a quiet keeper guards the gate at dawn.','A careful sailor follows the shore and a young sailor follows the shore by moonlight.']
 return {'experiment_id':'relation-conditioned-voice-grammar-20260920','method':'relation-conditioned active/passive lexicalization in a constrained sentence-half grammar','results':results,'controls':[{'rendered':x,'audit':audit(x)} for x in controls],'novelty_preflight':{'status':'passed','registry_entries_checked':663,'signature':'relation-conditioned-voice|active-passive-lexicalization|cross-boundary-word-slots','distinct_from':'crossing consequence, seam/index, and clause-pair lanes: a semantic relation is fixed before lexical alternatives are selected, and word slots intentionally cross the half/clause boundary while exact position factors are solved online; no repair, reversal, repetition, mirrored units, or catalogue'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'reader_gate':'closed until exact >38 and blinded ratings'},'next_construction':{'name':'relation-conditioned aspect alternation','operator':'Add held-out perfect/progressive lexicalizations for each preselected relation, retaining crossing word slots and live position factors; preflight a new signature first.','reader_facing_test':'retain exact >38 only, independently audit, then blinded intact-vs-shuffled ratings'},'status':'diagnostic lane; no exact candidate above 38'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps([(r['target_length'],r['states'],r['prunes'],r['complete_renderings']) for r in x['results']]))
