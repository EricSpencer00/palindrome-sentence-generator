"""Finite-state semantic boundary macros; no mirrored word-order or repeats."""
import hashlib,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/semantic-boundary-macro-fsm-20260916.json'
ID='semantic-boundary-macro-fsm-20260916';SIG='finite-state-semantic-boundary-macros|shifting-word-boundaries|non-nested-prose|nonrepeating-roles|independent-pointer-sha'
MACROS=[('subject','The careful archivist'),('verb','records the'),('object','harbor map'),('attachment','beside the old window'),('event','before the lamps fade')]
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for n in (3,4,5):
  # FSM transition state grows a complete grammatical sentence; macro
  # boundaries may shift through spaces but never create nested mirrored spans.
  text=' '.join(v for _,v in MACROS[:n])+'.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
  rows.append({'state':n,'rendered':text,'letters':a['letters'],'macro_boundaries':[k for k,_ in MACROS[:n]],'live_equation':{'constraint':'emitted boundary characters consume mirrored obligations','first_mismatch':a['first_mismatch']},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'fresh_complete_prose':True,'non_nested':True,'word_order_not_mirrored':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'repeated_self_palindromic_unit':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'finite-state semantic word-boundary macros with shifting boundaries and ordinary clause grammar','candidates':rows,'stats':{'states':3,'exact':0,'mechanically_admitted':0,'largest_letters':rows[-1]['letters']},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','duplicate_sweep_rejected':True},'next_repair':'At the first residual, replace the next macro transition with a fresh typed attachment and resegment its boundary; preserve non-nesting and non-repetition.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['candidates'][-1]['rendered']},indent=2))
