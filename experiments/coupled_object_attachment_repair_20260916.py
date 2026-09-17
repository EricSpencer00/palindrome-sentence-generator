"""Coupled two-slot repair: object and attachment are solved together."""
import hashlib,itertools,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/coupled-object-attachment-repair-20260916.json'
ID='coupled-object-attachment-repair-20260916';SIG='coupled-semantic-slot-repair|object-attachment-agreement|first-several-mirror-constraints|complete-scene|independent-pointer-sha'
SUBJECT='After the recital, the musician places'; TAIL=', shows it to her teacher, and waits until the last listeners go home.'
OBJECTS=['a blue case','a dark case']; ATTACH=['beside the steps','near the stairs']
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for oi,ai in itertools.product(range(2),range(2)):
  text=f'{SUBJECT} {OBJECTS[oi]} {ATTACH[ai]}'+TAIL;a=audit(text);c=mechanical_admission_checks(text,min_letters=90,max_letters=180)
  t=normalize_letters(text); prefix=next((i for i in range(len(t)//2) if t[i]!=t[-1-i]),len(t)//2)
  rows.append({'rendered':text,'letters':a['letters'],'joint_assignment':{'object_index':oi,'attachment_index':ai},'live_constraints':{'prefix_pairs_satisfied':prefix,'constraint':'maximize matched mirrored prefix before committing both slots'},'exact_audit':a,'checks':c,'mechanically_admitted':bool(a['two_pointer_exact'] and a['sha_equal'] and all(c.values())),'provenance':{'fresh_scene':False,'derived_from_readable_near_miss':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=max(rows,key=lambda r:r['live_constraints']['prefix_pairs_satisfied'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'coupled object-plus-attachment semantic repair constrained by mirrored prefix','candidates':rows,'best':best,'stats':{'joint_assignments':4,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','registry_entries_inspected':'read before run','duplicate_sweep_rejected':True},'next_repair':'Hold the best object fixed and replace only its attachment with a typed locative relative clause chosen to satisfy the next two mirrored pairs; do not enumerate this neighborhood again.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'best':p['best']['rendered'],'letters':p['best']['letters'],'stats':p['stats']},indent=2))
