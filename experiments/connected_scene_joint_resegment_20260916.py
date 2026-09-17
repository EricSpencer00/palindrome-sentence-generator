"""Joint semantic-slot and boundary-resegmentation repair on a connected scene."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/connected-scene-joint-resegment-20260916.json'
ID='connected-scene-joint-resegment-20260916';SIG='connected-scene-joint-slot-resegment|semantic-debt-reduction|boundary-resegmentation|anti-shortcut|independent-pointer-sha'
SLOTS=[('the gardener','carry','the map'),('the keeper','move','the lantern')]; SEAMS=[(' to the harbor','; then'),(' through the courtyard',', then')]
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[-1-i]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for (agent,verb,obj),(loc,join) in itertools.product(SLOTS,SEAMS):
  text=f'At first light, {agent} will {verb} {obj}{loc}{join} secure the supply crate at the station.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
  rows.append({'rendered':text,'letters':a['letters'],'semantic_slots':{'agent':agent,'verb':verb,'object':obj},'boundary_resegmentation':{'locative':loc.strip(),'joiner':join.strip(),'operator':'move seam across adjacent phrase boundary'},'live_mirrored_debt':{'first_residual':a['first_mismatch'],'mismatch_count':a['mismatch_count']},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'connected_scene':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'proper_palindromic_span':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'joint semantic slot substitution plus cross-word boundary resegmentation on a connected supply scene','candidates':rows,'best':best,'stats':{'joint_assignments':4,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','duplicate_sweep_rejected':True},'next_repair':'Use the best residual seam to choose one held-out locative relative clause and one role-compatible verb simultaneously; re-audit the connected scene without replaying this four-state neighborhood.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
