"""Fifth directed adjunct repair after the residual e/g obligation."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters
EXPERIMENT_ID="centerout-grammar-boundary-repair5-20260916"
SIGNATURE="center-out-grammar-state|residual-third-boundary-repair|heldout-scene-adjunct-authorship|event-preserving-valency|independent-exact-hash-audit"
OUT=ROOT/"runs"/f"{EXPERIMENT_ID}.json";PARENT=ROOT/"runs/centerout-grammar-boundary-repair4-20260916.json"
def audit(t):
 s=normalize_letters(t);m=[];i=0;j=len(s)-1
 while i<j:
  if s[i]!=s[j]:m.append({'left':i,'right':j,'a':s[i],'b':s[j]})
  i+=1;j-=1
 hf=hashlib.sha256(s.encode()).hexdigest();hr=hashlib.sha256(s[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_sha256','letters':len(s),'two_pointer_exact':bool(s) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':hf,'sha256_reverse':hr,'sha_equal':hf==hr}
def run():
 p=json.loads(PARENT.read_text());base=p['candidate']['rendered'];old='near the lamplit school at the height';new='near the lamplit school at the quiet height';assert old in base
 rendered=base.replace(old,new,1);a=audit(rendered);checks=mechanical_admission_checks(rendered,min_letters=39,max_letters=220)
 row={'label':'heldout-third-boundary-quiet-height-repair','rendered':rendered,'letters':a['letters'],'exact_audit':a,'checks':checks,'mechanically_admitted':bool(a['two_pointer_exact'] and a['sha_equal'] and all(checks.values())),'repair':{'parent_run':str(PARENT.relative_to(ROOT)),'changed_span':{'from':old,'to':new},'target_obligation':'residual e/g after matched th','changed_component':'right adjunct only'},'provenance':{'authored_event_preserved':True,'catalogue_text_copied':False,'finished_sentence_reversed':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False}}
 return {'experiment_id':EXPERIMENT_ID,'signature':SIGNATURE,'status':'completed','method':'single authored adjunct expansion aimed at the residual third mirrored character; all other event and grammar slots fixed','candidate':row,'stats':{'candidates':1,'exact':int(a['two_pointer_exact']),'mechanically_admitted':int(row['mechanically_admitted'])},'next_repair':'change the construction operator to a paired subject/adjunct boundary CSP because ordinary English adjuncts cannot freely realize the required three-letter suffix; preserve this failure evidence','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p,indent=2))
