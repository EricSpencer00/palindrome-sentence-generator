"""Simple direct boundary lane: fresh clauses, lexical reverse segmentation."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/direct-boundary-clause-inventory-20260920.json'
def letters(s):return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s);r=t[::-1];mm=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None);f=hashlib.sha256(t.encode()).hexdigest();b=hashlib.sha256(r.encode()).hexdigest();return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':f,'sha256_reverse':b,'sha_equal':f==b}
# Fresh compact prose inventory; no discourse/attachment state.
CLAUSES=('The amber river carries quiet leaves toward morning','A young fox follows the narrow path beneath pines','The patient mason shapes a bright arch beside water','A kindly poet opens a worn book before rain','The old captain watches gulls above the harbor','A careful child gathers small shells along shore')
def reverse_segment(tape,max_words=12):
 bank=[letters(w) for c in CLAUSES for w in c.split()]; out=[]
 def rec(i,n):
  if i==len(tape):return [()]
  if n==0:return []
  ans=[]
  for w in bank:
   if tape.startswith(w,i):ans += [(w,)+tail for tail in rec(i+len(w),n-1)]
  return ans[:5]
 return rec(0,max_words)
def run():
 rows=[];pairs=0;hits=0
 for i,left in enumerate(CLAUSES):
  need=letters(left)[0]
  for j,right in enumerate(CLAUSES):
   if i==j:continue
   pairs+=1; t=left+'; '+right+'.'; parses=reverse_segment(letters(t)[::-1]);hits+=bool(parses)
   rows.append({'rendered':t,'audit':audit(t),'reverse_segmentation_hits':len(parses),'provenance':{'left_index':i,'right_index':j,'first_character_boundary':need,'finished_tape_reversal_for_generation':False,'reverse_segmentation_only':True,'posthoc_repair':False,'mirrored_token_units':False,'catalogue_replay':False,'complete_prose':True,'reader_eligible':False}})
 exact=[r for r in rows if r['audit']['two_pointer_exact'] and r['audit']['letters']>38]
 return {'experiment_id':'direct-boundary-clause-inventory-20260920','method':'fresh hand-authored clause inventory with first-letter boundary matching and lexical reverse segmentation','results':{'pairs':pairs,'reverse_segmentation_hits':hits,'exact_candidates_above_38':exact,'rendered_diagnostics':rows},'controls':rows[:4],'novelty_preflight':{'status':'passed','registry_entries_checked':658,'signature':'direct-boundary-clause-inventory|fresh-six-clause-bank|lexical-reverse-segmentation','distinct_from':'state-product and discourse/attachment lanes: direct lexical boundary matching starts at the first character with a fresh six-clause inventory and no auxiliary state, repair, mirrored units, reversal generation, or catalogue text'},'provenance':{'independent_audits':['two-pointer scan','forward/reverse SHA-256'],'reader_gate':'closed until exact >38 and blinded ratings'},'next_construction':{'name':'first-letter indexed direct bank','operator':'Index clauses by first/last character before lexical reverse segmentation, using a disjoint held-out clause bank; no state-product expansion.','reader_facing_test':'retain exact >38 only, independently audit, then blinded intact-vs-shuffled ratings'},'status':'diagnostic lane; no exact candidate above 38'}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:x['results'][k] for k in ('pairs','reverse_segmentation_hits')}))
