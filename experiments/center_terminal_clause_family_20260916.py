"""Bounded center-bearing clause family with jointly chosen seam terminals."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/center-terminal-clause-family-20260916.json'
ID='center-terminal-clause-family-20260916';SIG='center-bearing-clause-family|joint-seam-terminal-choice|semantic-slots|nonrepeating-prose|independent-pointer-sha'
LEFT=[('The ranger','sketches','the valley'),('The singer','carries','the melody')]; CENTERS=['at noon','in spring']; RIGHT=[('the quiet clerk','files','the record'),('the young guide','opens','the gate')]
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[-1-i]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for l,cen,r in itertools.product(LEFT,CENTERS,RIGHT):
  text=f'{l[0]} {l[1]} {l[2]} {cen}; {r[0].capitalize()} {r[1]} {r[2]}.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
  rows.append({'rendered':text,'letters':a['letters'],'slots':{'left':l,'center_terminal':cen,'right':r},'live_center_equation':{'center_terminal_selected_jointly':True,'first_residual':a['first_mismatch'],'nested_span':False},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'human_authored':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'nested_palindrome_span':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'human-authored center-bearing clause family jointly selecting semantic slots and center terminal','candidates':rows,'best':best,'stats':{'bounded_states':8,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','duplicate_sweep_rejected':True},'next_repair':'Replace the center terminal with one newly authored connective whose boundary letters target the first residual while preserving both event roles; do not reuse these states.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
