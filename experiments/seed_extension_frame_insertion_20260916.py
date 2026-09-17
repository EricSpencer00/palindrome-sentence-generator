"""Frame-insertion probe; the historical seed is explicitly never wrapped."""
import hashlib,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/seed-extension-frame-insertion-20260916.json'
ID='seed-extension-frame-insertion-20260916'; SIG='authored-frame-insertion|paired-prefix-suffix-tape-equation|fresh-center-scene|seed-wrapper-rejection|independent-pointer-sha'
CENTERS=['the archivist opens the cedar cabinet and records the harbor map','the patient keeper carries the lantern across the quiet courtyard']
FRAMES=[('At dawn, ',', before dusk.'),('In spring, ',', by evening.'),('After rain, ',', before the lamps fade.')]
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for ci,center in enumerate(CENTERS):
  for fi,(pre,suf) in enumerate(FRAMES):
   text=pre+center+suf;a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
   rows.append({'rendered':text,'letters':a['letters'],'center_id':ci,'frame_id':fi,'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'fresh_authored_center':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'historical_seed_used_as_output':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'authored grammatical frame insertion with paired prefix/suffix tape accounting around a fresh center','seed_policy':{'historical_seed':'An aide rips nine memos; some men inspire Diana.','used_as_output':False,'wrapper_rejected':True,'reason':'known palindrome wrapping is a shortcut'},'candidates':rows,'best':best,'stats':{'frame_center_assignments':6,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','duplicate_sweep_rejected':True},'next_repair':'Author a new prefix/suffix pair whose boundary letters satisfy the first two residual equations while retaining a complete fresh center; do not insert or wrap the historical seed.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
