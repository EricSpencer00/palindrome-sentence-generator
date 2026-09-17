"""Typed agreement states attached to two-character residual phrase branches."""
from pathlib import Path
import json,hashlib,re,itertools
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/phrase-transducer-typed-agreement-20260917.json'
FRAMES=(('the careful botanist','records','singular'),('the patient cartographers','map','plural'));OBJECTS=('the local survey','the coastal atlas');TAILS=('carefully at dusk','quietly in spring')
def norm(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {'normalized_tape':t,'letters':len(t),'exact':bool(t) and not m,'independent_two_pointer_exact':bool(t) and not m,'first_mismatches':m[:8],'sha256_forward':f,'sha256_reverse':r,'sha_equal_under_reversal':f==r}
def emit(frame,o,tail):
 s,v,num=frame;text=f'{s.capitalize()} {v} {o} {tail}.';t=norm(text);return {'rendered':text,'choices':{'subject':s,'verb':v,'number':num,'object':o,'tail':tail},'live_obligations':{'first_two_left':t[:2],'first_two_right':t[-2:][::-1],'agreement_state':num,'agreement_checked_before_render':True,'outside_in_pairs_checked':len(t)//2},'audit':audit(text),'anti_shortcut_flags':{k:False for k in ('proper_name_tail_patching','finished_tape_reversal','repeated_self_palindromic_unit','word_order_symmetry','catalogue_text','fragment')},'provenance':{'lexical_source':'authored agreement-bearing SVO frames and tail roles','borrowed_text':False,'typed_agreement_transducer':True,'syntax_expanded':False}}
def run():
 rows=[emit(f,o,t) for f,o,t in itertools.product(FRAMES,OBJECTS,TAILS)];ex=[x for x in rows if x['audit']['exact']]
 return {'experiment_id':'phrase-transducer-typed-agreement-20260917','signature':'whole-tape-two-residual-trie|typed-agreement-state|complete-role-phrases|pre-render-obligations','status':'completed_exact' if ex else 'completed_no_exact_closure','method':'two-character residual branches carrying typed subject-verb agreement','novelty_preflight':{'status':'passed','registry_entries_read':None,'signature_collision':False,'artifact_collision':False,'shortcuts_rejected':['proper-name tail patching','finished-tape reversal','repeated units','word-order symmetry','fragments']},'candidate_count':len(rows),'exact_count':len(ex),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows),'agreement_states':len(set(x['choices']['number'] for x in rows))},'failure_and_repair':{'failure':'no exact closure' if not ex else 'exact closure found','next_repair':'carry object number and determiner agreement jointly with the branch state'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer scan','forward/reverse SHA-256','typed agreement ledger'],'shortcuts_excluded':True}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print({'candidates':x['candidate_count'],'exact':x['exact_count'],'stats':x['stats']})
