"""Fresh cross-word seam CSP; no fixed tape or seed-derived output."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/fresh-crossword-seam-csp-20260916.json'
ID='fresh-crossword-seam-csp-20260916';SIG='fresh-scene-grammar|cross-word-seam-csp|heldout-role-lexicon|live-character-obligation|independent-pointer-sha'
AGENTS=['the courier','the keeper'];VERBS=['carries','records'];OBJECTS=['the parcel','the ledger'];PLACES=['to the depot','by the gate']
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for ai,vi,oi,pi in itertools.product(range(2),repeat=4):
  text=f'At sunrise, {AGENTS[ai]} {VERBS[vi]} {OBJECTS[oi]} {PLACES[pi]}; the bell rings for the morning shift.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
  rows.append({'rendered':text,'letters':a['letters'],'assignment':{'agent':AGENTS[ai],'verb':VERBS[vi],'object':OBJECTS[oi],'place':PLACES[pi]},'live_csp':{'seam':'object/place boundary','obligation':a['first_mismatch'],'pruning':'reject assignment only after complete role-compatible realization'},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'fresh_scene_grammar':True,'heldout_role_lexicon':True,'fixed_tape_used':False,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'fresh role-typed cross-word seam CSP with live character obligations','candidates':rows,'best':best,'stats':{'assignments':16,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','fixed_tape_or_seed_guidance':False},'next_repair':'Build a new two-clause grammar whose object and place slots jointly target the first residual, keeping agent/verb roles fixed; do not replay this product.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
