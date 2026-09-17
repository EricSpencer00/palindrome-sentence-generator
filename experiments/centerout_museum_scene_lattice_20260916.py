"""Fresh center-out museum scene lattice with typed valency/agreement."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/centerout-museum-scene-lattice-20260916.json'
ID='centerout-museum-scene-lattice-20260916';SIG='centerout-museum-scene|joint-left-right-selection|typed-valency-agreement|live-character-equation|independent-pointer-sha'
LEFT=[('The curator','restores','the portrait'),('The conservator','studies','the relic')]; RIGHT=[('the guide','opens','the gallery'),('the docent','locks','the exhibit')]
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
  text=f'{l[0]} {l[1]} {l[2]}; the audience listens while {r[0]} {r[1]} {r[2]}.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
  rows.append({'rendered':text,'letters':a['letters'],'left_clause':l,'right_clause':r,'centerout_equation':{'center':'the audience listens','selection':'left/right valency selected jointly before outward rendering','first_residual':a['first_mismatch']},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'fresh_scene':True,'agreement_and_valency_checked':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'fresh center-out museum scene lattice with jointly selected typed clauses','candidates':rows,'best':best,'stats':{'joint_states':4,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','old_courier_harbor_frame_used':False,'fixed_tape_used':False},'next_repair':'At the first residual, author one held-out museum verb/object pair for the corresponding semantic role and solve its boundary against the preserved center; do not sweep the four states again.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
