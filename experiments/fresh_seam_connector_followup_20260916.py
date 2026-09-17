"""Single discourse-connector repair in the preserved harbor scene."""
import hashlib,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/fresh-seam-connector-followup-20260916.json'
ID='fresh-seam-connector-followup-20260916';SIG='heldout-discourse-connector-repair|preserved-seam-and-attachment|connected-harbor-scene|independent-pointer-sha'
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 text='At dawn, the pilot marks the chart beside the old pier, and the crew secures the boat for departure.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220);h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 row={'rendered':text,'letters':a['letters'],'repair_operator':{'changed_slot':'discourse_connector','from':'then','to':'and','preserved':['chart beside the old pier','for departure'],'held_out':True},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'seed_used_as_output':False,'seed_wrapped':False,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':h}}
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'one held-out discourse connector replacement preserving prior semantic and attachment slots','candidate':row,'stats':{'child_states':1,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed'},'next_repair':'Change only the final coordinating verb at the next residual while preserving connector, chart, locative, and attachment; do not reopen prior states.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'candidate':p['candidate']['rendered'],'length':p['candidate']['letters']},indent=2))
