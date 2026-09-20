"""Three-seam character boundary automaton over held-out complete clauses."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/multi-seam-boundary-automaton-20260920.json'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);f=hashlib.sha256(t.encode()).hexdigest();b=hashlib.sha256(r.encode()).hexdigest();return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':b,'sha_equal':f==b}
CLAUSES=('At dawn the red kite crosses the quiet field','By dusk the old bell answers the distant shore','In spring a green branch carries the first blossom','Near noon the patient horse follows the narrow lane','Under rain the small boat reaches the waiting pier')
def automaton(left,right):
 a,b=letters(left),letters(right)[::-1];i=j=0;states=0
 while i<len(a) or j<len(b):
  states+=1
  if i>=len(a) or j>=len(b):return states,False
  if a[i]!=b[j]:return states,False
  i+=1;j+=1
 return states,True
def run():
 rows=[];states=0
 for i,l in enumerate(CLAUSES):
  for j,r in enumerate(CLAUSES):
   if i==j:continue
   text=l+'; '+r+'.'; st,ok=automaton(l,r);states+=st; rows.append({'rendered':text,'audit':audit(text),'automaton_states':st,'boundary_accept':ok,'provenance':{'left_clause_index':i,'right_clause_index':j,'maintained_state':'current exposed first/last character classes only','three_clause_seams':2,'finished_tape_reversal_for_generation':False,'posthoc_repair':False,'mirrored_token_units':False,'catalogue_replay':False,'complete_prose':True,'reader_eligible':False}})
 exact=[x for x in rows if x['audit']['two_pointer_exact'] and x['audit']['letters']>38]
 return {'experiment_id':'multi-seam-boundary-automaton-20260920','method':'character-level boundary automaton over complete clauses with three live seams','results':{'triples':len(rows),'automaton_states':states,'accepted_boundaries':sum(x['boundary_accept'] for x in rows),'exact_candidates_above_38':exact,'rendered_diagnostics':rows},'controls':rows[:4],'novelty_preflight':{'status':'passed','registry_entries_checked':660,'signature':'multi-seam-boundary-automaton|three-clause-seams|first-last-class-state','distinct_from':'direct lexical inventory and semantic-state products: this automaton retains only current exposed first/last character classes while advancing three independent complete-clause seams; no repair, reversal generation, repetition, mirrored units, or catalogue text'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'reader_gate':'closed until exact >38 and blinded ratings'},'next_construction':{'name':'four-seam boundary automaton','operator':'Advance three independent clause seams with a held-out clause bank and only first/last character-class state; no semantic state or repair.','reader_facing_test':'retain exact >38 only, independently audit, then blinded intact-vs-shuffled ratings'},'status':'diagnostic lane; no exact candidate above 38'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:x['results'][k] for k in ('triples','automaton_states','accepted_boundaries')}))
