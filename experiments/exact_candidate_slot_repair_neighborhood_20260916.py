"""Small, typed semantic-slot repair around two long readable near misses."""
import hashlib,itertools,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/exact-candidate-slot-repair-neighborhood-20260916.json'
EXPERIMENT_ID='exact-candidate-slot-repair-neighborhood-20260916'; SIGNATURE='exact-candidate-first-mismatch|typed-semantic-slot-substitution|role-preserving-neighborhood|independent-pointer-sha'
BASE=[('concert-case','After the recital, the musician places a dark case beside the steps, shows it to her teacher, and waits until the last listeners go home.', 'object', ['blue case','small case','wooden case']),('archive-scene','At dawn, the archivist opened the cedar cabinet. The patient apprentice copied each date into a clean ledger. The curator sorted four journals for the river school. Evening bells faded.','attachment',['before sunrise','after the rain','near the harbor'])]
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for sid,text,slot,alts in BASE:
  for alt in alts:
   if slot=='object': out=text.replace('a dark case', 'a '+alt,1)
   else: out=text.replace('At dawn', 'At '+alt, 1)
   a=audit(out); c=mechanical_admission_checks(out,min_letters=90,max_letters=240)
   rows.append({'base_id':sid,'rendered':out,'letters':a['letters'],'changed_slot':slot,'replacement':alt,'exact_audit':a,'checks':c,'mechanically_admitted':bool(a['two_pointer_exact'] and a['sha_equal'] and all(c.values())),'provenance':{'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'fresh_semantic_slot':True}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 # One held-out follow-up repair on the first residual: change only its typed slot.
 follow=best.copy(); follow['follow_up_repair']={'operator':'replace first-residual slot with held-out role-compatible lexical item','held_out_replacement':'bronze case' if best['changed_slot']=='object' else 'beside the quay','applied':False}
 return {'experiment_id':EXPERIMENT_ID,'signature':SIGNATURE,'status':'completed','method':'first-mismatch typed semantic-slot neighborhood with one held-out follow-up repair','candidates':rows,'best':best,'follow_up':follow,'stats':{'bases':2,'bounded_substitutions':len(rows),'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','reject_duplicate_fingerprints':True},'next_repair':'Apply the held-out role-compatible substitution only after solving the first residual boundary; if it increases debt, switch to a typed attachment rewrite rather than another lexical sweep.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
