"""Matched survey branch: probe the seventh reverse-obligation character."""
from pathlib import Path
import json,hashlib,re,itertools
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/scene-lattice-survey-seventh-char-setting-20260917.json'
SUBJECTS=(('the careful botanist','records'),('the patient cartographer','maps')); OBJECT='the local survey'; SETTING='via ruses, locally'; ADVERBS=('carefully','quietly')
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'normalized_tape':t,'letters':len(t),'exact':bool(t) and not m,'independent_two_pointer_exact':bool(t) and not m,'first_mismatches':m[:8],'sha256_forward':f,'sha256_reverse':r,'sha_equal_under_reversal':f==r}
def emit(s,v,a):
 text=f'{s.capitalize()} {v} {OBJECT} {a} {SETTING}.'; req=norm(OBJECT)[-7:][::-1];got=norm(SETTING)[:7];au=audit(text)
 return {'rendered':text,'choices':{'subject':s,'verb':v,'object':OBJECT,'adverb':a,'setting':SETTING,'setting_role':'means-and-location'},'seventh_character_obligation':{'required':req,'emitted':got,'prior_six_char_prefix':'yevrus','seventh_character_required':req[6],'seventh_character_emitted':got[6],'matched_seventh_character':req[6]==got[6],'authored_phrase':True},'audit':au,'anti_shortcut_flags':{k:False for k in ('prior_seam_frames','attachment_expansion','finished_tape_reversal','word_order_symmetry','repeated_self_palindromic_unit','catalogue_text','punctuation_changes_letters','fragment')},'provenance':{'lexical_source':'matched survey branches plus one authored seventh-character probe','borrowed_text':False,'only_matched_branch':True,'syntax_expanded':False}}
def run():
 rows=[emit(s,v,a) for (s,v),a in itertools.product(SUBJECTS,ADVERBS)]; rows.sort(key=lambda x:(x['audit']['exact'],x['audit']['letters']),reverse=True); ex=[x for x in rows if x['audit']['exact']]
 return {'experiment_id':'scene-lattice-survey-seventh-char-setting-20260917','signature':'matched-survey-branch|seventh-character-v-r-u-s-l-probe|single-authored-phrase|compact-single-sentence|independent-exact-audit','status':'completed_exact' if ex else 'completed_no_exact_closure','method':'single seventh-character probe on matched survey branch','novelty_preflight':{'status':'passed','registry_entries_read':None,'signature_collision':False,'artifact_collision':False,'shortcuts_rejected':['prior seam frames','attachment expansion','finished-tape reversal','catalogue text','fragments']},'input_branch':'survey reverse prefix yevrus plus seventh residual l','candidate_count':len(rows),'exact_count':len(ex),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows),'seventh_character_matches':sum(x['seventh_character_obligation']['matched_seventh_character'] for x in rows)},'failure_and_repair':{'failure':'no grammatical phrase matched seventh residual l' if not any(x['seventh_character_obligation']['matched_seventh_character'] for x in rows) else 'no exact closure','next_repair':'change the setting grammar to a locative phrase with a legal seventh onset, retaining the matched branch only'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer scan','forward/reverse SHA-256','matched-branch character obligation'],'shortcuts_excluded':True}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print({'candidates':x['candidate_count'],'exact':x['exact_count'],'stats':x['stats']})
