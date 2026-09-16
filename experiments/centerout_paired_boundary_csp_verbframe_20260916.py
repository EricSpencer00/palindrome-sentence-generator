"""Fresh paired-boundary CSP with a second ordinary verb frame."""
from __future__ import annotations
import hashlib,json,sys,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters
EXPERIMENT_ID="centerout-paired-boundary-csp-verbframe-20260916"
SIGNATURE="paired-subject-adjunct-boundary-csp|fresh-verb-frame|ordinary-svo-grammar|live-character-obligation|independent-exact-hash-audit"
OUT=ROOT/"runs"/f"{EXPERIMENT_ID}.json"
SUBJECTS=["the steady mason","the watchful clerk"]; FRAMES=[("seals","the parcel"),("records","the invoice")]
ADJUNCTS=["beside the riverside depot at noon","under the copper awning before rain"]
def audit(t):
 s=normalize_letters(t);m=[];i=0;j=len(s)-1
 while i<j:
  if s[i]!=s[j]:m.append({'left':i,'right':j,'a':s[i],'b':s[j]})
  i+=1;j-=1
 hf=hashlib.sha256(s.encode()).hexdigest();hr=hashlib.sha256(s[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_sha256','letters':len(s),'two_pointer_exact':bool(s) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':hf,'sha256_reverse':hr,'sha_equal':hf==hr}
def run():
 choices=[]
 for s,(v,o),a in itertools.product(SUBJECTS,FRAMES,ADJUNCTS):
  text=f"{s} {v} {o}. {a}.";x=audit(text);choices.append((x['mismatch_count'],s,v,o,a,text,x))
 _,s,v,o,a,text,x=min(choices,key=lambda q:q[0]);checks=mechanical_admission_checks(text,min_letters=39,max_letters=220)
 row={'label':'fresh-verb-frame-paired-csp','rendered':text,'letters':x['letters'],'exact_audit':x,'checks':checks,'mechanically_admitted':bool(x['two_pointer_exact'] and x['sha_equal'] and all(checks.values())),'csp':{'fresh_domains':{'subjects':SUBJECTS,'frames':FRAMES,'adjuncts':ADJUNCTS},'selected':{'subject':s,'verb':v,'object':o,'adjunct':a},'objective':'minimize mirrored boundary mismatch count'},'provenance':{'authored_event_preserved':True,'ordinary_svo_order':True,'fresh_lexical_domains':True,'catalogue_text_copied':False,'finished_sentence_reversed':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False}}
 return {'experiment_id':EXPERIMENT_ID,'signature':SIGNATURE,'status':'completed','method':'paired boundary CSP with fresh subject, verb, object, and adjunct domains; second ordinary verb frame added','candidate':row,'stats':{'bounded_assignments':8,'candidates':1,'exact':int(x['two_pointer_exact']),'mechanically_admitted':int(row['mechanically_admitted'])},'next_repair':'add a fresh transitive frame with a tense/agreement state and carry the selected boundary obligations forward; no replay','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p,indent=2))
