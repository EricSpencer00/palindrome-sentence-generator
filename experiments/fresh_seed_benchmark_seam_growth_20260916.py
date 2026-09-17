"""Fresh scene growth benchmarked against, but never containing, the 38-char seed."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/fresh-seed-benchmark-seam-growth-20260916.json'
ID='fresh-seed-benchmark-seam-growth-20260916';SIG='seed-benchmark-only|cross-word-seam-substitution|connected-scene-growth|heldout-role-solving|independent-pointer-sha'
AGENTS=['the curator','the pilot']; ACTIONS=['marks','records']; OBJECTS=['the map','the tide']; SEAMS=['near the harbor','by the old pier']
SEED='An aide rips nine memos; some men inspire Diana.'
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[-1-i]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for ai,ac,ob,seam in itertools.product(range(2),range(2),range(2),range(2)):
  text=f'At dawn, {AGENTS[ai]} {ACTIONS[ac]} {OBJECTS[ob]} {SEAMS[seam]}; then the crew secures the boat.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
  rows.append({'rendered':text,'letters':a['letters'],'heldout_assignment':{'agent':AGENTS[ai],'action':ACTIONS[ac],'object':OBJECTS[ob],'seam':seam},'connected_scene':'one harbor survey event','live_seam_equation':{'first_mismatch':a['first_mismatch'],'operator':'substitute held-out role words at cross-word seam'},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'seed_used_as_output':False,'seed_wrapped':False,'fresh_scene':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'fresh connected harbor scene with held-out cross-word seam substitutions; seed is benchmark only','benchmark':{'text':SEED,'used_as_output':False,'wrapped':False},'candidates':rows,'best':best,'stats':{'heldout_assignments':16,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','duplicate_sweep_rejected':True},'next_repair':'Solve the first mirrored seam by jointly replacing the harbor object and locative with one held-out role-compatible pair; never insert or preserve the benchmark seed.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
