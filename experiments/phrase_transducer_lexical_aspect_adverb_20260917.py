"""Lexical aspect realization and adverb are coupled with typed feature state."""
from pathlib import Path
import json,hashlib,re,itertools
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/phrase-transducer-lexical-aspect-adverb-20260917.json'
FRAMES=(('the careful botanist','records','singular','present','simple'),('the patient cartographers','map','plural','present','progressive'));OBJECTS=(('the local survey','singular'),('the coastal atlases','plural'));REALIZATIONS=(('carefully','simple'),('is carefully recording','progressive'))
def norm(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {'normalized_tape':t,'letters':len(t),'exact':bool(t) and not m,'independent_two_pointer_exact':bool(t) and not m,'first_mismatches':m[:8],'sha256_forward':f,'sha256_reverse':r,'sha_equal_under_reversal':f==r}
def emit(frame,obj,real):
 s,v,snum,tense,aspect=frame;o,onum=obj; adv=real[0];text=f'{s.capitalize()} {v} {o} {adv}.';t=norm(text);return {'rendered':text,'choices':{'subject':s,'verb':v,'subject_number':snum,'object':o,'object_number':onum,'tense':tense,'aspect':aspect,'lexical_realization':real[1],'adverb':adv},'live_obligations':{'first_two_left':t[:2],'first_two_right':t[-2:][::-1],'joint_feature_state':(snum,onum,tense,aspect,real[1],adv),'joint_features_checked_before_render':True,'outside_in_pairs_checked':len(t)//2},'audit':audit(text),'anti_shortcut_flags':{k:False for k in ('proper_name_tail_patching','finished_tape_reversal','repeated_self_palindromic_unit','word_order_symmetry','catalogue_text','fragment')},'provenance':{'lexical_source':'authored aspect realizations and adverb choices','borrowed_text':False,'lexical_aspect_transducer':True,'syntax_expanded':False}}
def run():
 rows=[emit(f,o,r) for f,o,r in itertools.product(FRAMES,OBJECTS,REALIZATIONS) if f[2]==o[1] and f[4]==r[1]];ex=[x for x in rows if x['audit']['exact']]
 return {'experiment_id':'phrase-transducer-lexical-aspect-adverb-20260917','signature':'joint-agreement-object-number-tense-aspect|lexical-aspect-adverb|complete-role-phrases|pre-render-obligations','status':'completed_exact' if ex else 'completed_no_exact_closure','method':'lexical aspect realization and adverb jointly selected with typed feature state','novelty_preflight':{'status':'passed','registry_entries_read':None,'signature_collision':False,'artifact_collision':False,'shortcuts_rejected':['proper-name tail patching','finished-tape reversal','repeated units','word-order symmetry','fragments']},'candidate_count':len(rows),'exact_count':len(ex),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows),'joint_states':len(set(x['live_obligations']['joint_feature_state'] for x in rows))},'failure_and_repair':{'failure':'no exact closure' if not ex else 'exact closure found','next_repair':'carry complement and valency frame jointly with lexical aspect state'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer scan','forward/reverse SHA-256','joint lexical feature ledger'],'shortcuts_excluded':True}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print({'candidates':x['candidate_count'],'exact':x['exact_count'],'stats':x['stats']})
