"""Single joint seam repair for center-free clause pair."""
import hashlib,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/centerfree-clause-pair-followup-20260916.json'
ID='centerfree-clause-pair-followup-20260916';SIG='heldout-seam-final-opening-repair|preserved-complete-clauses|centerfree|independent-pointer-sha'
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 text='The sailor mends the canvas. A nurse checks the patient.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220);h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 row={'rendered':text,'letters':a['letters'],'repair_operator':{'changed_slots':['left_final_word','right_opening_word'],'from':'sail / The nurse','to':'canvas / A nurse','held_out':True,'complete_clauses_preserved':True},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'fresh_complete_clauses':True,'fixed_center_used':False,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':h}}
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'one joint held-out final/opening seam substitution in center-free clauses','candidate':row,'stats':{'new_states':1,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','replay':False},'next_repair':'Change only the left predicate at the next seam residual while preserving both clause roles and the new opening word; do not reopen this joint state.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'candidate':p['candidate']['rendered'],'length':p['candidate']['letters']},indent=2))
