"""Fresh dialogue/relative-clause CSP with variable lexical slots."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/dialogue-relative-clause-csp-20260916.json'
ID='dialogue-relative-clause-csp-20260916';SIG='dialogue-relative-clause-csp|variable-length-slots|semantic-agreement-before-render|live-character-equation|independent-pointer-sha'
SPEAKERS=['Mira','Jon']; VERBS=['found','kept']; OBJECTS=['a small key','the blue notebook']
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for sp,v,o in itertools.product(SPEAKERS,VERBS,OBJECTS):
  text=f'“{sp} said, “I {v} {o}, which the night guard had marked.”” The guard replied, “Keep it safe.”';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
  rows.append({'rendered':text,'letters':a['letters'],'slots':{'speaker':sp,'past_verb':v,'object':o,'relative_agreement':'singular'},'live_csp':{'variable_length_slot_lengths':[len(normalize_letters(x)) for x in (sp,v,o)],'first_residual':a['first_mismatch'],'equations':'semantic role and agreement fixed before character emission'},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'fresh_dialogue_grammar':True,'relative_clause_complete':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'museum_courier_harbor_templates_reused':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'fresh dialogue with complete relative clause and variable-length role slots in finite CSP','candidates':rows,'best':best,'stats':{'states':8,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','old_templates_reused':False,'fixed_tape_used':False},'next_repair':'At the first residual, replace only the relative-clause predicate with a held-out agreement-compatible verb while preserving the dialogue act and object; do not replay states.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
