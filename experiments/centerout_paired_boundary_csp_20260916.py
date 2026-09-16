"""Bounded paired subject/adjunct boundary CSP after lexical repair failure."""
from __future__ import annotations
import hashlib,json,sys,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters
EXPERIMENT_ID="centerout-paired-boundary-csp-20260916"
SIGNATURE="paired-subject-adjunct-boundary-csp|ordinary-svo-grammar|live-character-obligation|event-preserving-valency|bounded-single-construction|independent-exact-hash-audit"
OUT=ROOT/"runs"/f"{EXPERIMENT_ID}.json"
SUBJECTS=["the careful porter","the gray porter"]
ADJUNCTS=["near the lamplit school at the quiet height","near the lamplit school beside the quiet gate"]
def audit(t):
 s=normalize_letters(t);m=[];i=0;j=len(s)-1
 while i<j:
  if s[i]!=s[j]:m.append({'left':i,'right':j,'a':s[i],'b':s[j]})
  i+=1;j-=1
 hf=hashlib.sha256(s.encode()).hexdigest();hr=hashlib.sha256(s[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_sha256','letters':len(s),'two_pointer_exact':bool(s) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':hf,'sha256_reverse':hr,'sha_equal':hf==hr}
def run():
 # The CSP evaluates one bounded paired assignment, retaining ordinary SVO order.
 pairs=list(itertools.product(SUBJECTS,ADJUNCTS)); scored=[]
 for subject,adj in pairs:
  t=f"{subject} delivers the sealed parcel. {adj}.";a=audit(t);scored.append((a['mismatch_count'],subject,adj,t,a))
 _,subject,adj,rendered,a=min(scored,key=lambda x:x[0]);checks=mechanical_admission_checks(rendered,min_letters=39,max_letters=220)
 row={'label':'bounded-paired-subject-adjunct-csp','rendered':rendered,'letters':a['letters'],'exact_audit':a,'checks':checks,'mechanically_admitted':bool(a['two_pointer_exact'] and a['sha_equal'] and all(checks.values())),'csp':{'domains':{'subject':SUBJECTS,'adjunct':ADJUNCTS},'selected_subject':subject,'selected_adjunct':adj,'objective':'minimize mirrored boundary mismatch count under ordinary grammar'},'provenance':{'authored_event_preserved':True,'ordinary_svo_order':True,'catalogue_text_copied':False,'finished_sentence_reversed':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False}}
 return {'experiment_id':EXPERIMENT_ID,'signature':SIGNATURE,'status':'completed','method':'bounded paired subject/adjunct CSP with live character obligations and ordinary SVO valency','candidate':row,'stats':{'bounded_assignments':len(pairs),'candidates':1,'exact':int(a['two_pointer_exact']),'mechanically_admitted':int(row['mechanically_admitted'])},'next_repair':'expand the CSP with a second ordinary verb frame while retaining paired boundary obligations; do not replay the same lexical domains','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p,indent=2))
