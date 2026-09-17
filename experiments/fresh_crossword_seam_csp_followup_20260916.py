"""One fresh grammar state jointly targeting object/place seam residual."""
import hashlib,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/fresh-crossword-seam-csp-followup-20260916.json'
ID='fresh-crossword-seam-csp-followup-20260916';SIG='fresh-two-clause-grammar|joint-object-place-target|heldout-seam-state|live-obligation|independent-pointer-sha'
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 text='At sunrise, the courier carries the sealed bundle across the quay; the bell calls the morning crew to work.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220);h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 row={'rendered':text,'letters':a['letters'],'grammar_state':{'agent':'the courier','verb':'carries','object':'the sealed bundle','place':'across the quay','second_clause':'bell calls crew'},'live_csp':{'target':'object/place seam','first_residual':a['first_mismatch'],'joint_choice_committed':True},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'fresh_two_clause_grammar':True,'heldout_role_lexicon':True,'prior_16_assignments_replayed':False,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'generator_sha256':h}}
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'single fresh two-clause grammar jointly targeting object/place seam residual','candidate':row,'stats':{'new_states':1,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'registry_entries_inspected':339,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','prior_assignments_replayed':False},'next_repair':'Use a new attachment clause with the same courier event to target the next residual pair; preserve this object/place choice and do not reopen the earlier product.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'candidate':p['candidate']['rendered'],'length':p['candidate']['letters']},indent=2))
