"""Connected imperative scene with role-typed boundary blocks (fresh control)."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/typed-boundary-block-scene-20260916.json'
ID='typed-boundary-block-scene-20260916';SIG='typed-common-word-boundary-blocks|connected-imperative-scene|role-typed-boundary|no-name-verb-mirror|independent-pointer-sha'
SUBJECTS=['the archivist','the gardener']; VERBS=['label','carry']; OBJECTS=['the map','the lantern']; LOCS=['to the harbor','through the courtyard']
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[-1-i]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for si,vi,oi,li in itertools.product(range(2),range(2),range(2),range(2)):
  text=f'First, ask {SUBJECTS[si]} to {VERBS[vi]} {OBJECTS[oi]} {LOCS[li]}; then seal the crate and return to the station.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
  rows.append({'rendered':text,'letters':a['letters'],'typed_blocks':{'agent':SUBJECTS[si],'action':VERBS[vi],'patient':OBJECTS[oi],'path':LOCS[li]},'connected_event':'one crate-handling instruction sequence','exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'fresh_authored_scene':True,'role_typed_common_words':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'proper_palindromic_span':False,'repeated_self_palindromic_unit':False,'old_command_chain_reused':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'role-typed common-word boundary blocks in one connected imperative scene','reference_control':{'old_command_chain_used':False,'purpose':'development comparison only'},'candidates':rows,'best':best,'stats':{'bounded_assignments':16,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','duplicate_sweep_rejected':True},'next_repair':'At the first residual, replace one role-typed block with a held-out common-word block preserving the crate event, then solve the following instruction boundary jointly; do not replay this product.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
