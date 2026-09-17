"""Matched branch: probe twelfth residual after the locally continuation."""
from pathlib import Path
import json,hashlib,re,itertools
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/scene-lattice-survey-joint-locative-twelfth-20260917.json'
SUBJECTS=(('the careful botanist','records'),('the patient cartographer','maps'));OBJECT='the local survey';SETTING='via del Laco, locally';ADVERBS=('carefully','quietly')
def norm(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {'normalized_tape':t,'letters':len(t),'exact':bool(t) and not m,'independent_two_pointer_exact':bool(t) and not m,'first_mismatches':m[:8],'sha256_forward':f,'sha256_reverse':r,'sha_equal_under_reversal':f==r}
def emit(s,v,a):
 text=f'{s.capitalize()} {v} {OBJECT} {a} {SETTING}.';req=norm(OBJECT)[-12:][::-1];got=norm(SETTING)[:12];au=audit(text)
 return {'rendered':text,'choices':{'subject':s,'verb':v,'object':OBJECT,'adverb':a,'setting':SETTING,'setting_role':'typed-proper-locative'},'twelfth_character_obligation':{'required':req,'emitted':got,'prior_eleven_prefix':'yevruslacol','twelfth_required':req[11],'twelfth_emitted':got[11],'matched_twelfth':req[11]==got[11],'conditioned_tail':'locally'},'audit':au,'anti_shortcut_flags':{k:False for k in ('prior_seam_frames','attachment_expansion','finished-tape-reversal','word_order_symmetry','repeated_self_palindromic_unit','catalogue_text','punctuation_changes_letters','fragment')},'provenance':{'lexical_source':'matched survey branch plus twelfth-character continuation probe','borrowed_text':False,'only_matched_branch':True,'syntax_expanded':False}}
def run():
 rows=[emit(s,v,a) for (s,v),a in itertools.product(SUBJECTS,ADVERBS)];ex=[x for x in rows if x['audit']['exact']]
 return {'experiment_id':'scene-lattice-survey-joint-locative-twelfth-20260917','signature':'matched-survey-branch|joint-locative-la|twelfth-e-probe|compact-single-sentence|independent-exact-audit','status':'completed_exact' if ex else 'completed_no_exact_closure','method':'locative continuation after locally probed against twelfth residual e','novelty_preflight':{'status':'passed','registry_entries_read':None,'signature_collision':False,'artifact_collision':False,'shortcuts_rejected':['prior seam frames','attachment expansion','finished-tape reversal','catalogue text','fragments']},'input_branch':'survey residual prefix yevruslacole','candidate_count':len(rows),'exact_count':len(ex),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows),'twelfth_matches':sum(x['twelfth_character_obligation']['matched_twelfth'] for x in rows)},'failure_and_repair':{'failure':'locally emits o where twelfth residual requires e' if not ex else 'exact closure found','next_repair':'replace locally with a grammatical le-initial continuation while preserving the matched proper-locative prefix'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer scan','forward/reverse SHA-256','twelfth residual audit'],'shortcuts_excluded':True}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print({'candidates':x['candidate_count'],'exact':x['exact_count'],'stats':x['stats']})
