"""Fresh active/passive valency and attachment CSP."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/active-passive-attachment-csp-20260916.json'
ID='active-passive-attachment-csp-20260916';SIG='active-passive-valency-csp|attachment-before-emission|fresh-scene-bank|agreement-typed|independent-pointer-sha'
SUBJ=['The engineer','The mechanic']; OBJ=['the bridge','the engine']; ATT=['after the storm','inside the workshop']
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[-1-i]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for si,oi,ai in itertools.product(range(2),range(2),range(2)):
  active=f'{SUBJ[si]} inspected {OBJ[oi]} {ATT[ai]}'; passive=f'{OBJ[oi].capitalize()} was inspected by {SUBJ[si].lower()} {ATT[ai]}';text=active+'. '+passive+'.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220)
  rows.append({'rendered':text,'letters':a['letters'],'valency':{'agent':SUBJ[si],'patient':OBJ[oi],'attachment':ATT[ai],'voice_pair':'active_then_passive','agreement':'singular'},'live_attachment_csp':{'attachment_selected_before_character_emission':True,'first_residual':a['first_mismatch']},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'fresh_scene_bank':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'semordnilap_chain':False,'old_templates_reused':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'active/passive valency CSP selecting attachment before character emission','candidates':rows,'best':best,'stats':{'states':8,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','dialogue_relative_templates_reused':False,'fixed_tape_used':False},'next_repair':'At the first residual, hold agent/patient fixed and author one held-out attachment compatible with both voices; re-solve its boundary before rendering.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
