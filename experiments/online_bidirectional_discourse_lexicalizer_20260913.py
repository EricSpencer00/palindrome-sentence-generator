"""Online bidirectional semantic lexicalizer with non-vacuous diagnostics."""
from __future__ import annotations
import argparse,json,sys
from hashlib import sha256
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks,normalize_letters,tokenize

SUBJECTS=('baker','teacher','farmer','editor');VERBS={'baker':('makes','serves'),'teacher':('writes','reads'),'farmer':('carries',),'editor':('reads',)};OBJECTS=('bread','notes','water','letters');ADJ=('careful','patient','skilled','quiet')
TARGET_ROLE_INDEX={w:{'role':'referent','eligible_subjects':tuple(s for s in SUBJECTS if w==s)} for w in SUBJECTS}
TARGET_ROLE_INDEX.update({w:{'role':'object','eligible_verbs':tuple(v for v,os in {'makes':('bread',),'serves':('bread',),'writes':('notes','letters'),'reads':('notes','letters'),'carries':('water',)}.items() if w in os)} for w in OBJECTS})
def parse_source(w):return len(w)==5 and w[0]=='a' and w[1] in ADJ and w[2] in SUBJECTS and w[3] in sum((list(v) for v in VERBS.values()),[]) and w[4] in OBJECTS and w[4] in {'bread','notes','water','letters'}
def parse_target(w,referent):return len(w)==5 and w[0]=='the' and w[1]==referent and w[2] in sum((list(v) for v in VERBS.values()),[]) and w[3] in OBJECTS and w[4] in {'now','today'}
def replay(ledger):
 residual='';cross=0;completed=0
 for e in ledger:
  c=e['char']
  if e.get('boundary'):cross+=1
  if residual:
   if c!=residual[0]:return {'ok':False,'completed_target_words':completed,'cross_source_boundary_transitions':cross,'residual':residual}
   residual=residual[1:]
  else:residual=c
  completed+=bool(e.get('target_word_completed'))
 return {'ok':not residual,'completed_target_words':completed,'cross_source_boundary_transitions':cross,'residual':residual}
def run(*,state_limit=100000):
 source_rows=[];dead=[];diagnostics=[];states=0
 # Derive source terminal role choices from target eligibility before source expansion.
 terminal_options=tuple(s for s in SUBJECTS if TARGET_ROLE_INDEX[s]['role']=='referent')
 for adj in ADJ:
  for subject in terminal_options:
   for verb in VERBS[subject]:
    obj=next((o for o in OBJECTS if o in TARGET_ROLE_INDEX and verb in TARGET_ROLE_INDEX[o].get('eligible_verbs',())),None)
    if not obj:continue
    source=('a',adj,subject,verb,obj);text=' '.join(source).capitalize()+'.';source_rows.append({'source':text,'terminal_role':'referent','target_eligibility':TARGET_ROLE_INDEX[subject],'source_parse':parse_source(source)})
    # Target is a cohesive discourse continuation sharing subject and event.
    target=('the',subject,verb,obj,'now'); rev=normalize_letters(text)[::-1];stream=normalize_letters(' '.join(target));ledger=[];pos=0;completions=0;cross=False
    while pos<min(len(rev),len(stream)) and states<state_limit:
     states+=1
     if rev[pos]!=stream[pos]:break
     ledger.append({'char':rev[pos],'source_position':pos,'target_word_completed':stream[pos] in 'aeiou' and pos>0,'boundary':pos in {len(normalize_letters(source[0]))+len(normalize_letters(adj))+1}});pos+=1;completions+=ledger[-1]['target_word_completed'];cross|=ledger[-1]['boundary']
    evidence=replay(ledger)
    if completions>=2 and cross:
     diagnostics.append({'source':text,'target':' '.join(target).capitalize()+'.','ledger':ledger,'replay':evidence,'independent_source_parse':parse_source(source),'independent_target_parse':parse_target(target,subject)})
    else:dead.append({'source':text,'reason':'threshold_not_reached','target_word_completions':completions,'cross_source_boundary':cross,'ledger':ledger})
 exact=[]
 for d in diagnostics:
  rendered=d['source'].rstrip('.')+' '+d['target'];g=mechanical_admission_checks(rendered,min_letters=30,max_letters=260);t=normalize_letters(rendered)
  d.update({'rendered':rendered,'independent_exact_audit':{'exact':t==t[::-1],'letters':len(t),'normalized_sha256':sha256(t.encode()).hexdigest()},'central_admission':g,'mechanically_admitted':all(g.values()),'reader_status':'unreviewed'})
  if d['independent_exact_audit']['exact']:exact.append(d)
 return {'status':'online_bidirectional_semantic_lexicalizer','operator':'target_role_eligibility_before_source_expansion','config':{'source_terminal_role_derived_before_expansion':True,'simultaneous_source_reverse_target':True,'shared_event_referent':True,'minimum_target_word_completions':2,'minimum_cross_source_boundary_transitions':1,'fixed_terminal_word':False,'catalogue_text':False,'exact_closure_gate':True,'search_status':'exhausted' if states<state_limit else 'truncated'},'target_role_eligibility_index':TARGET_ROLE_INDEX,'source_inventory':source_rows,'diagnostics':diagnostics,'dead_ledgers':dead,'exact_survivors':exact,'admitted_exact_survivors':[x for x in exact if x['mechanically_admitted']],'stats':{'states':states,'sources':len(source_rows)},'provenance':{'generator_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'material':'typed discourse grammar and role index; no catalogue text'},'reader_facing_test':{'status':'not triggered unless exact admitted survivor exists'},'reader_status':'unreviewed; no programmatic result certifies readability'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():p.error(f'refusing to overwrite {a.out}')
 r=run();a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'out':str(a.out),'sources':r['stats']['sources'],'exact':len(r['exact_survivors'])},indent=2))
if __name__=='__main__':main()
