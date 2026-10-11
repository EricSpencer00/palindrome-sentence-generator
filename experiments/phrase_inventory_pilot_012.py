"""Bounded phrase inventory API fixture with finite full enumeration control."""
import hashlib,itertools,json,gzip,time,zipfile,math
from dataclasses import asdict
from pathlib import Path
from llm_palindrome.phrase_inventory_search import Phrase,PhraseInventoryDAG
from llm_palindrome.bidirectional_lexical import exact_grammar_palindromes
from llm_palindrome.admission import normalize_letters
from llm_palindrome.phrase_api_loop import repetition_features
OUT=Path('research/block-seams/phrase-inventory-012')
def fixture():
 inv=[];plans=[]
 def slot(values,a,b,role,source,bindings=None):
  ids=[]
  for text in values:
   pid='f'+str(len(inv)).zfill(3);inv.append(Phrase(pid,text,a,b,role,source,tuple((bindings or {}).items())));ids.append(pid)
  return ids
 def plan(pid,method,states,groups):plans.append(dict(id=pid,method=method,states=states,slots=groups))
 # One syntactic question with two NP alternatives, not two sentence mirrors.
 q=slot(['Was it '],'START','Q','question_subject_copula','native scene003/abcba008 partial phrase')
 np=slot(['a car','a cat','a war','a wart','a rat','a drawer','desserts','evil'],'Q','NP','question_predicate_nominal','mixed existing native vocabulary; finite authored role alternatives')
 con=slot([' or '],'NP','ALT','alternative_connector','native car/cat question fragment')
 np2=slot(['a car','a cat','a war','a wart','a rat','a drawer','desserts','evil'],'ALT','REL','second_predicate_nominal','same finite authored role alternatives')
 rel=slot([' I saw?'],'REL','END','postposed_relative_observation','native scene003 final relative phrase',{'observer':'I'})
 plan('single_question','phrase_bridge_overhang',['START','Q','NP','ALT','REL','END'],[q,np,con,np2,rel])
 # Partial word overhang: noun car/cat is split at ca / r or t. The right
 # seam's role completes the adjective/noun unit, not a free filler syllable.
 a=slot(['Was it a ca'],'START','PARTIAL_CA','question_partial_nominal','native scene003 word-level derivation')
 b=slot(['r or a ca','t or a ca'],'PARTIAL_CA','PARTIAL_CA_2','noun_completion_and_alternative','car/cat lexical alternatives; copied control structure')
 c=slot(['r I saw?','t I saw?'],'PARTIAL_CA_2','END','noun_completion_and_relative','car/cat lexical alternatives; copied control structure',{'observer':'I'})
 plan('partial_word_question','phrase_bridge_overhang',['START','PARTIAL_CA','PARTIAL_CA_2','END'],[a,b,c])
 # Native reaction vocabulary; mixed object/adjective pairs searched by exact
 # debt. Adjacent clause roles preserve observer identity explicitly.
 obs=slot(['I saw '],'START','OBS','observation_subject_verb','supplement004-006',{'observer':'I'})
 obj=slot(['desserts','evil','war','a car','a cat'],'OBS','REACTION','observation_theme','supplement004-006 plus native scene003 objects')
 reaction=slot(['; stressed was ','; live was ','; raw was '],'REACTION','PRED','reaction_predicate_inversion','supplement004-006; inversion is quality concern')
 subj=slot(['I.','Mom.'],'PRED','END','reaction_subject','I from supplement; Mom mismatch diagnostic',{'observer':'I'})
 # Separate mismatch fragment tests actor consistency, not just noun sharing.
 mom=slot(['Mom.'],'PRED','END','reaction_subject','native Mom vocabulary; deliberately conflicting observer',{'observer':'Mom'})
 plan('observation_reaction','non_ABCBA_role_scene',['START','OBS','REACTION','PRED','END'],[obs,obj,reaction,[subj[0]]+mom])
 # Single comparative clause. This is an existing copied calibration control.
 ma=slot(['Ma is as '],'START','COMP','comparative_subject','frozen009 grammar')
 adj=slot(['selfless','evil','stressed','raw','live'],'COMP','DEG','adjective_predicate','existing grammar/native vocabulary')
 end=slot([' as I am.'],'DEG','END','degree_comparison','frozen009 grammar')
 plan('comparative','non_ABCBA_role_scene',['START','COMP','DEG','END'],[ma,adj,end])
 # Phrase ABCBA slot arm, compatible reverse debt and explicit inter-word seam.
 # Existing Eve/refer phrase supplies a calibration; no whole-sentence bank.
 e=slot(['Eve, '],'START','REFER_PREFIX','human_topic','existing typed vocabulary')
 re=slot(['re'],'REFER_PREFIX','VERB_MID','partial_referral_verb','existing refer lemma')
 f=slot(['f'],'VERB_MID','VERB_SUFFIX','partial_referral_center','existing refer lemma')
 er=slot(['er '],'VERB_SUFFIX','THEME','referral_verb_completion','existing refer lemma')
 eve=slot(['Eve.'],'THEME','END','human_theme','existing typed vocabulary')
 plan('phrase_ABCBA_referral','phrase_ABCBA',['START','REFER_PREFIX','VERB_MID','VERB_SUFFIX','THEME','END'],[e,re,f,er,eve])
 return inv,plans

def run():
 OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();inv,plans=fixture();g=PhraseInventoryDAG(inv,plans)
 def save(n,o):(OUT/n).write_text(json.dumps(o,indent=2)+'\n')
 native=[];import_receipts=[]
 for path in sorted((OUT/'imports').glob('*.zip')):
  with zipfile.ZipFile(path) as z:
   assert z.testzip() is None
   rawname=next(n for n in z.namelist() if n.endswith('integration/raw.jsonl'));rows=[json.loads(s) for s in z.read(rawname).decode().splitlines()];native+=rows
   receipt=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/assembly-receipt.json'))));import_receipts.append(dict(local_zip=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),receipt=receipt))
 save('native-import-receipts.json',import_receipts)
 save('plan.json',dict(seed='phrase-inventory-012',max_product_operations=100000,max_product_seconds=5,max_accepted_paths=2000,max_enumeration_paths=2000,plans=len(plans),models=0,no_sentence_bank=True,inventory_source='authored calibration fragments from durably imported native proposals; not newly LLM-generated fragment inventory',stopping='one complete fixture search and independent enumeration; no adaptive batches',sampling='ceil10% per method x whole-tape-in-native/frozen009 flag; raw duplicate derivations retained',quality='unrated; copied and short cases remain negative target outcomes'))
 save('inventory.json',[asdict(p) for p in inv]);save('plans.json',plans);save('fragment-api-contract.json',dict(task='Propose useful partial grammatical phrases rather than whole palindromes; deterministic solver handles reverse tape debt.',required=['id','text','entry','exit','role','source','bindings'],max_fragments=48,forbidden_input='Do not supply completed paragraphs or mirrored sentence pairs as one fragment.',states='Use named obligations START/Q/NP/ALT/REL/END or propose explicit plan transitions; every phrase entry/exit must match a plan slot.',bindings='Actor/discourse variables name exact entity values; solver reports conflicts. Do not infer progression from shared noun.',partial_words='Provide lexical lemma, consumed offset, and completion obligation for split-word fragments; no filler debt repair.',source='Disclose stock or copied fragments; whole-copy match remains an outcome feature.',objective='Meaningful phrase syntax and consistent scene; soft repetition and progression costs, not quality guarantees.',budget='One native parent batch<=48fragments,<=8plans; CPU compilation enforces finite input and resource limits; no paidAPI.' ))
 with gzip.open(OUT/'product-states.jsonl.gz','wt') as trace:paths,receipt=exact_grammar_palindromes(g,max_work=100000,max_paths=2000,seconds=5,trace=lambda r:trace.write(json.dumps(r)+'\n'))
 raw=[g.materialize(path) for path in paths];enum=[];enum_failures=0;enum_paths=0;enum_proposals=[]
 for fi,p in enumerate(plans):
  for ids in itertools.product(*p['slots']):
   enum_paths+=1;assert enum_paths<=2000
   ps=[g.inventory[i] for i in ids];text=''.join(x.text for x in ps);t=normalize_letters(text)
   enum_proposals.append(dict(plan=p['id'],phrase_ids=ids,text=text,tape=t,exact=t==t[::-1]))
   if t!=t[::-1]:enum_failures+=1;continue
   bindings={};ok=True
   for x in ps:
    for k,v in x.bindings:
     if k in bindings and bindings[k]!=v:ok=False
     bindings[k]=v
   enum.append((tuple(ids),t,ok))
 save('enumeration-control-all-proposals.json',enum_proposals)
 assert sorted((tuple(r['phrase_ids']),r['tape'],r['role_consistent']) for r in raw)==sorted(enum)
 old={normalize_letters(x['text']) for x in native}
 import gzip as gz
 frozen={json.loads(s)['tape'] for s in gz.open('research/block-seams/structural-breadth-009/all-deduplicated-outputs.jsonl.gz','rt')}
 for r in raw:
  r['id']='phrase012-'+hashlib.sha256((r['plan']+'|'+r['tape']).encode()).hexdigest()[:16];r['in_native_whole_tapes']=r['tape'] in old;r['in_frozen009']=r['tape'] in frozen;r['new_relative_to_inputs']=r['tape'] not in old|frozen;r['repetition']=repetition_features(r['text']);r['whole_copy_disclosed']=r['in_native_whole_tapes'];r['human_label']=None
  # Penalize explicit clausal inversion and repeat fractions, never outlaw.
  r['soft_diagnostic_cost']=r['repetition']['token_repeat_fraction']+.5*r['repetition']['trigram_repeat_fraction']+(1 if r['plan']=='observation_reaction' else 0)
  r['quality_unrated']=True
 save('raw-outputs.json',raw);save('search-receipt.json',receipt)
 save('graph.json',dict(start=g.start,accept=g.accept,arcs=[asdict(a) for a in g.arcs],epsilon={str(k):sorted(v) for k,v in g.eps.items() if v}))
 accepted=[r for r in raw if r['role_consistent']];unique={r['tape']:r for r in accepted};save('unique-outputs.json',sorted(unique.values(),key=lambda r:(r['soft_diagnostic_cost'],r['id'])))
 from collections import defaultdict
 strata=defaultdict(list)
 for r in raw:strata[(r['method'],r['in_native_whole_tapes'] or r['in_frozen009'])].append(r)
 sample=[];manifest=[]
 for st,rows in sorted(strata.items()):
  k=math.ceil(len(rows)/10);chosen=sorted(rows,key=lambda r:hashlib.sha256(('phrase012sample'+r['id']).encode()).hexdigest())[:k];sample+=chosen;manifest.append(dict(stratum=st,population=len(rows),sample=k,ids=[r['id'] for r in chosen]))
 save('sample.json',sample);save('sample-manifest.json',manifest)
 summary=dict(fragment_count=len(inv),plans=len(plans),enumerated_attempts=enum_paths,enumerated_exactness_failures=enum_failures,product_accepted_derivations=len(raw),role_consistent_derivations=len(accepted),role_conflicts=len(raw)-len(accepted),unique_exact_outputs=len(unique),duplicate_derivations=len(accepted)-len(unique),new_unique_to_native_and009=sum(r['new_relative_to_inputs'] for r in unique.values()),outputs_ge60=sum(r['letters']>=60 for r in unique.values()),max_letters=max(r['letters'] for r in raw),product_matches_independent_enumeration=True,product_work=receipt['work'],partial_phrase_states=receipt['partial_word_states'],sample_occurrences=len(sample),sample_denominator=len(raw),sample_percent=100*len(sample)/len(raw),elapsed_seconds=time.monotonic()-start,quality_scores=0,human_approvals=0,target_success='none: no>=60letteroutput; phrase-level mechanics work but vocabulary/grammar coverage is limiting')
 save('summary.json',summary);save('checkpoint.json',dict(status='complete_bounded_negative_pilot',summary=summary,next='Parent native Luna proposes role-compatible fragments to this API, not whole palindromes; use new vocabulary and meaningful grammar obligations before expanding length.',human010_feedback='pending, not inferred'))
 print(json.dumps(summary,indent=2));print('\n'.join(r['text'] for r in unique.values()))
if __name__=='__main__':run()
