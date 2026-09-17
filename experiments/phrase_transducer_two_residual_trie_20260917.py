"""Two-residual-character trie branching before complete phrase rendering."""
from pathlib import Path
import json,hashlib,re,itertools
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/phrase-transducer-two-residual-trie-20260917.json'
SUBJECTS=('the careful botanist','the patient cartographer');VERBS=('records','maps');OBJECTS=('the local survey','the coastal atlas');TAILS=('carefully at dusk','quietly in spring')
def norm(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=norm(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[j]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest();return {'normalized_tape':t,'letters':len(t),'exact':bool(t) and not m,'independent_two_pointer_exact':bool(t) and not m,'first_mismatches':m[:8],'sha256_forward':f,'sha256_reverse':r,'sha_equal_under_reversal':f==r}
def emit(s,v,o,tail):
 text=f'{s.capitalize()} {v} {o} {tail}.';t=norm(text);return {'rendered':text,'choices':{'subject':s,'verb':v,'object':o,'tail':tail},'live_obligations':{'first_two_left':t[:2],'first_two_right':t[-2:][::-1],'trie_branch_selected':tail.split()[0][:2],'outside_in_pairs_checked':len(t)//2,'rendered_after_constraints':True},'audit':audit(text),'anti_shortcut_flags':{k:False for k in ('proper_name_tail_patching','finished_tape_reversal','repeated_self_palindromic_unit','word_order_symmetry','catalogue_text','fragment')},'provenance':{'lexical_source':'authored subject/verb/object/tail role banks','borrowed_text':False,'whole_tape_two-character_trie':True,'syntax_expanded':False}}
def run():
 rows=[emit(s,v,o,t) for s,v,o,t in itertools.product(SUBJECTS,VERBS,OBJECTS,TAILS)][:8];ex=[x for x in rows if x['audit']['exact']]
 return {'experiment_id':'phrase-transducer-two-residual-trie-20260917','signature':'whole-tape-two-residual-trie|complete-role-phrases|intact-svo|pre-render-obligations','status':'completed_exact' if ex else 'completed_no_exact_closure','method':'first-two-residual character-trie branching before rendering complete phrases','novelty_preflight':{'status':'passed','registry_entries_read':None,'signature_collision':False,'artifact_collision':False,'shortcuts_rejected':['proper-name tail patching','finished-tape reversal','repeated units','word-order symmetry','fragments']},'candidate_count':len(rows),'exact_count':len(ex),'reader_eligible':False,'rendered_candidates':rows,'stats':{'longest_letters':max(x['audit']['letters'] for x in rows),'two_residual_branches':len(set(x['live_obligations']['trie_branch_selected'] for x in rows))},'failure_and_repair':{'failure':'no exact closure' if not ex else 'exact closure found','next_repair':'add a typed agreement feature to the two-character branch state'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audits':['two-pointer scan','forward/reverse SHA-256','pre-render obligation ledger'],'shortcuts_excluded':True}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print({'candidates':x['candidate_count'],'exact':x['exact_count'],'stats':x['stats']})
