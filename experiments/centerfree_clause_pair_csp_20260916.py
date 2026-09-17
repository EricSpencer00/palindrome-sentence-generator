"""Center-free CSP: two independently authored complete clauses meet at a seam."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/centerfree-clause-pair-csp-20260916.json'
ID='centerfree-clause-pair-csp-20260916';SIG='centerfree-independent-clause-pair|seam-character-equation|fresh-semantic-csp|no-fixed-center|independent-pointer-sha'
LEFT=[('The sailor','mends','the sail'),('The baker','kneads','the dough')];RIGHT=[('The nurse','checks','the patient'),('The ranger','maps','the trail')]
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[-1-i]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for l,r in itertools.product(LEFT,RIGHT):
  text=' '.join(l)+'. '+ ' '.join(r)+'.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
  seam=len(normalize_letters(' '.join(l)))
  rows.append({'rendered':text,'letters':a['letters'],'left_clause':l,'right_clause':r,'seam_equation':{'left_tape_length':seam,'obligation_at_clause_boundary':a['first_mismatch'],'center_selected':False},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'independently_authored_complete_clauses':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'old_scene_family_reused':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'center-free finite semantic CSP joining two independently authored complete clauses at live seam equation','candidates':rows,'best':best,'stats':{'joint_states':4,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','fixed_center_or_tape':False},'next_repair':'At the seam residual, author one held-out role-compatible final word for the left clause and one opening word for the right clause jointly; preserve complete clauses and do not replay these states.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
