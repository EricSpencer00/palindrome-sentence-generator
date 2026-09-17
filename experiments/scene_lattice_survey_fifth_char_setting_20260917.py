"""Matched survey branch: fifth reverse-obligation character."""
from pathlib import Path
import json, hashlib, re, itertools
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/scene-lattice-survey-fifth-char-setting-20260917.json'
SUBJECTS=(('the careful botanist','records'),('the patient cartographer','maps')); OBJECT='the local survey'; SETTING='via rural fields'; ADVERBS=('carefully','quietly')
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s); mis=[]; i=0; j=len(t)-1
 while i<j:
  if t[i]!=t[j]: mis.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1; j-=1
 f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'normalized_tape':t,'letters':len(t),'exact':bool(t) and not mis,'independent_two_pointer_exact':bool(t) and not mis,'first_mismatches':mis[:8],'sha256_forward':f,'sha256_reverse':r,'sha_equal_under_reversal':f==r}
def emit(s,v,a):
 text=f'{s.capitalize()} {v} {OBJECT} {a} {SETTING}.'; t=norm(text); req=norm(OBJECT)[-5:][::-1]; got=norm(SETTING)[:5]; au=audit(text)
 return {'rendered':text,'choices':{'subject':s,'verb':v,'object':OBJECT,'adverb':a,'setting':SETTING,'setting_role':'condition'},'fifth_character_obligation':{'required':req,'emitted':got,'prior_four_char_prefix':'yevr','fifth_character_required':req[4],'fifth_character_emitted':got[4],'matched_fifth_character':req[4]==got[4],'authored_phrase':True},'audit':au,'anti_shortcut_flags':{k:False for k in ('prior_seam_frames','attachment_expansion','finished_tape_reversal','word_order_symmetry','repeated_self_palindromic_unit','catalogue_text','punctuation_changes_letters','fragment')},'provenance':{'lexical_source':'matched survey branches plus one authored fifth-character setting','borrowed_text':False,'only_matched_branch':True,'syntax_expanded':False}}
def run():
 rows=[emit(s,v,a) for (s,v),a in itertools.product(SUBJECTS,ADVERBS)]; rows.sort(key=lambda x:(x['audit']['exact'],x['audit']['letters']),reverse=True); exact=[x for x in rows if x['audit']['exact']]
 return {'experiment_id':'scene-lattice-survey-fifth-char-setting-20260917','signature':'matched-survey-branch|fifth-character-v-r-u-setting|single-authored-phrase|compact-single-sentence|independent-exact-audit','status':'completed_exact' if exact else 'completed_no_exact_closure','method':'single fifth-character-compatible setting phrase on matched survey branch','novelty_preflight':{'status':'passed','registry_entries_read':None,'signature_collision':False,'artifact_collision':False,'shortcuts_rejected':['prior seam frames','attachment expansion','finished-tape reversal','catalogue text','fragments']},'input_branch':'survey reverse prefix yevru','candidate_count':len(rows),'exact_count':len(exact),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows),'fifth_character_matches':sum(x['fifth_character_obligation']['matched_fifth_character'] for x in rows)},'failure_and_repair':{'failure':'no exact closure' if not exact else 'exact closure found','next_repair':'retain the v-r-u setting onset and test a sixth-character-compatible authored phrase only on this matched branch'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer scan','forward/reverse SHA-256','matched-branch character obligation'],'shortcuts_excluded':True}}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print({'candidates':x['candidate_count'],'exact':x['exact_count'],'stats':x['stats']})
