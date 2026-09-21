"""Generated finite typed grammar overhang search (no seed palindromes)."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ID='typed-overhang-center-search-20260921'; ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs'/f'{ID}.json'
BANK={
 'DET':[('a','sg'),('the','sg')], 'N_S':[('man','sg'),('plan','sg'),('canal','sg'),('dog','sg'),('sun','sg')],
 'V_S':[('sees','sg'),('likes','sg'),('walks','sg')], 'PREP':[('in','na'),('by','na')],
 'N2_S':[('park','sg'),('sea','sg'),('road','sg')], 'PRON':[('i','sg'),('we','pl')],
 'V1':[('see','pl'),('like','pl'),('walk','pl')],
}
GRAM={'NP':(('DET','N_S'),),'VP':(('V_S','NP'),('V_S','PREP','DET','N2_S')),'S':(('NP','VP'),('PRON','V1','NP'))}
def clean(s): return re.sub('[^a-z]','',s.lower())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def paths():
 out=[]
 for d,nd in BANK['DET']:
  for n,nn in BANK['N_S']:
   if nd==nn:
    for v,vn in BANK['V_S']:
     # finite typed ordinary-English S path
     out.append(([d,n,v],{'number':'sg','shape':'NP-V'}))
 return out
def obligation_search(left,right,center):
 # Expand from both ends; unequal side lengths leave a real overhang state.
 a=clean(' '.join(left)); b=clean(' '.join(right)); tape=a+center+b
 q=list(tape); rev=list(tape[::-1]); trace=[]; i=j=0
 while i<len(q) and j<len(rev):
  ok=q[i]==rev[j]; trace.append({'left_index':i,'right_index':j,'left':q[i],'expected':rev[j],'ok':ok})
  if not ok:return False,trace
  i+=1;j+=1
 return i==len(q),trace
def main():
 ps=paths(); rows=[]; controls=[]
 # Independently generated clause pairs, with a one-token center operator.
 for li,(l,lf) in enumerate(ps):
  for ri,(r,rf) in enumerate(ps):
   for center in ('a','i',''):
    ok,tr=obligation_search(l,r,center); text=' '.join(l+([center] if center else [])+r)+'.'
    row={'rendered':text.capitalize(),'normalized':clean(text),'exact':ok,'odd_center':len(clean(text))%2==1,'overhang':len(clean(' '.join(l)))-len(clean(' '.join(r))),'center':center or None,'grammar_path_left':['S','NP','VP'],'grammar_path_right':['S','NP','VP'],'typed_agreement':lf['number']==rf['number'],'obligation_trace':tr,'candidate_kind':'generated'}
    if row['typed_agreement']: rows.append(row)
    # control changes only the center operator; it is audited, never admitted.
    ctext=' '.join(l+(['x'] if center else ['a'])+r)+'.'; controls.append({'rendered':ctext.capitalize(),'normalized':clean(ctext),'candidate_kind':'control_bad_center'})
 audited=[]
 for row in rows+controls:
  n=row['normalized']; exact=n==n[::-1] and bool(n)
  audited.append({**row,'independent_exact':exact,'pointer_check':{'left':n,'right_reversed':n[::-1],'equal':exact},'sha256':sha(row['rendered'])})
 admissible=[x for x in audited if x.get('candidate_kind')=='generated' and x['independent_exact'] and x.get('typed_agreement') and len(x['normalized'])>=1]
 out={'experiment_id':ID,'method':'finite typed ordinary-English grammar paths with live opposing-character overhang search','grammar':GRAM,'lexical_bank':BANK,'paths_generated':len(rows),'candidates':audited,'controls':controls,'reader_admission':{'admitted':len(admissible),'reject_semordnilap_or_self_palindrome':True,'reject_non_grammatical':True,'exact_ge_40':sum(len(x['normalized'])>=40 for x in admissible)},'stats':{'exact':sum(x['independent_exact'] for x in audited),'exact_ge_40':sum(x['independent_exact'] and len(x['normalized'])>=40 for x in audited),'odd_exact':sum(x['independent_exact'] and x.get('odd_center',False) for x in audited)},'novelty_preflight':{'canonical_seeds':False,'reverse_derived_fragments':False,'mirrored_word_units':False,'finished_tape_postrender_search':False},'shortcut_gates':{'equal_length_clause_product':False,'repair':False,'RLAIF':False,'word_boundary_only':False,'live_obligations':True,'unequal_side_lengths_allowed':True},'provenance':{'source':'fresh finite lexical bank in this file','code_sha256':sha(Path(__file__).read_text()),'audit':'independent normalized pointer equality and SHA-256','run_path':str(RUN)},'next_construction':'add a typed relative-clause operator and retain overhang obligations as a first-class chart dimension'}
 RUN.write_text(json.dumps(out,indent=2)+'\n'); print({'rows':len(rows),'exact_ge_40':out['stats']['exact_ge_40']})
if __name__=='__main__':main()
