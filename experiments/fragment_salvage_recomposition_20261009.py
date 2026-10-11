"""Finite source-fragment reconstruction, with semantic subject compatibility."""
import hashlib,itertools,json,math,re,time
from collections import defaultdict,Counter
from pathlib import Path
from experiments.derived_paragraph_controls_20261009 import UNITS
from experiments.block_seam_comparison_20261009 import render_paragraph
from llm_palindrome.admission import normalize_letters
from llm_palindrome.typed_constituents import TypedGrammar,words,NAMES
from llm_palindrome.paragraph_residual_join import exact_residual_pairs,JoinBudgetExceeded
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/block-seams'
NEW=('a man','bees','can','did','i','live','lives','no evil','on','saw','see')
HUMAN={'rider','clerk','child','artist','man','nurse','pilot','baker','gardener','sailor','writer','teacher','children','artists','men'}
ANIMAL={'dogs','bees','birds','bats','pets'}
PRONOUNS={'i','he','she','we','they','you'}
NAMES_SET={n.lower() for n in NAMES}
def animate(np):return bool(np) and (np[-1] in HUMAN|ANIMAL|NAMES_SET or np in {(p,) for p in PRONOUNS})
def source_spans(bank,piece):
 target=words(piece);out=[]
 for i,row in enumerate(bank):
  tape=words(row['text'])
  for at in range(len(tape)-len(target)+1):
   if tape[at:at+len(target)]==target:out.append({'source':'data/v3_bank.json#'+str(i),'kind':row['source'],'origin':row['origin'],'word_span':[at,at+len(target)]})
 return out

def enumerate_clauses(components,semantic):
 g=TypedGrammar(set(itertools.chain.from_iterable(words(c) for c in components)));clauses=set();blocked=0
 for slots,label in g.paths:
  choices=[tuple(t for t in slot if ' '.join(t) in components) for slot in slots]
  if not all(choices):continue
  for parts in itertools.product(*choices):
   if semantic and (':see' in label) and not animate(parts[0]):blocked+=1;continue
   clauses.add(tuple(itertools.chain.from_iterable(parts)))
 return g,sorted(clauses),blocked

def run():
 planpath=OUT/'fragment-salvage-recomposition-001-plan.json'
 if planpath.exists():raise RuntimeError('immutable run exists')
 inv=json.loads((OUT/'licensed-control-inventory-v2-001.json').read_text());bank=json.loads((ROOT/'data/v3_bank.json').read_text());feedback=json.loads((OUT/'human-five-output-feedback-001.json').read_text())
 provenance={c:source_spans(bank,c) for c in NEW};assert all(provenance.values())
 excluded=set(inv['excluded_normalized'])|{normalize_letters(r['text']) for r in bank}|{normalize_letters(r['original_text']) for r in feedback['whole_output_labels']}
 definitions=[('core_structural',set(UNITS),False),('core_animate_subjects',set(UNITS),True),('v3_fragment_salvage',set(UNITS)|set(NEW),True)]
 plan={'human_feedback_message':feedback['source_message'],'whole_outputs_bad':5,'fragments_human_approved':0,'source_bank_sha256':hashlib.sha256((ROOT/'data/v3_bank.json').read_bytes()).hexdigest(),'core_component_provenance':'Existing v2 inventory sources; supplied successful whole texts are excluded and not inventory inputs.','v3_components':provenance,'filler_exclusions':['association','tai','ita','cos','lac','til','no i'],'semantics':'For see-family predicates, subjects must have explicit person/animal/pronoun/name head; mail, iron and evil cannot see. Experimental semantic rule, not user-approved fragment judgments.','arms':[{'name':n,'components':sorted(c),'animate_see_subjects':s} for n,c,s in definitions],'sentence_counts':[2,3,4],'letter_band':[60,119],'equal_arm_bounds':{'deadline_seconds':20,'join_work':20000000},'workers':1,'seeds':'Complete deterministic finite enumeration, no stochastic generation seed; fixed SHA256 sampling seed salvage-sample-001. Arms share declared limits; complete-space counts differ and are reported.','limits':'Source-derived small lexical slice, not unrestricted English; no novelty or human-readability claim.','search_algorithm':'Exact residual joins; unequal clause-half lengths retained through palindromic middle debt; full existing exact/grammar/distinct/exclusion gates unchanged.'}
 planpath.write_text(json.dumps(plan,indent=2)+'\n');results=[]
 for name,components,semantic in definitions:
  start=time.monotonic();deadline=start+20;g,clauses,blocked=enumerate_clauses(components,semantic);norm={c:normalize_letters(' '.join(c)) for c in clauses};sequences=set();receipts=[];status='completed'
  for n in (2,3,4):
   a=n//2;b=n-a;left=list(itertools.product(clauses,repeat=a));right=list(itertools.product(clauses,repeat=b));lt=[''.join(norm[c] for c in s) for s in left];rt=[''.join(norm[c] for c in s) for s in right]
   try:pairs,receipt=exact_residual_pairs(lt,rt,max_work=20000000-sum(r['work'] for r in receipts),deadline=deadline)
   except JoinBudgetExceeded:status='truncated_join_budget';break
   receipts.append({'clause_count':n,**receipt,'complete_combination_denominator':len(clauses)**n})
   for i,j in pairs:
    seq=left[i]+right[j];t=''.join(norm[c] for c in seq)
    if 60<=len(t)<=119:sequences.add(seq)
  rows=[]
  for seq in sorted(sequences):
   text=render_paragraph(tuple((c,g.complete(c)) for c in seq));t=normalize_letters(text);assert t and t==t[::-1];assert g.text_paragraph(text,4)
   toks=words(text);filler=any(w in toks for w in ('association','tai','ita','cos','lac','til')) or any(toks[i:i+2]==('no','i') for i in range(len(toks)-1));assert not filler
   distinct=len(set(seq))==len(seq);excluded_whole=t in excluded;bad_subjects=[c for c in seq if 'sees' in c and not animate(c[:c.index('sees')])];assert not semantic or not bad_subjects
   predicates=tuple(w for c in seq for w in c if w in ('sees','see','saw','live','lives','did'))
   rows.append({'id':name+'-'+hashlib.sha256(text.encode()).hexdigest()[:16],'text':text,'exact':True,'letters':len(t),'clause_count':len(seq),'distinct_clauses':distinct,'known_or_supplied_whole':excluded_whole,'mechanically_eligible':distinct and not excluded_whole,'inanimate_sees_clauses':[' '.join(c) for c in bad_subjects],'predicate_types':sorted(set(predicates)),'known_example_component_derivation':True,'human_label':None,'coherence':'unreviewed; grammar and animate subjects are not discourse certification'})
  strata=defaultdict(list)
  for r in rows:strata[(r['clause_count'],'60-79' if r['letters']<80 else '80-119')].append(r)
  sample=[];manifest=[]
  for key,rr in sorted(strata.items()):
   count=math.ceil(len(rr)/10);chosen=sorted(rr,key=lambda r:hashlib.sha256(('salvage-sample-001|'+r['id']).encode()).hexdigest())[:count];sample.extend(chosen);manifest.append({'arm':name,'clauses':key[0],'band':key[1],'denominator':len(rr),'sample_count':count,'ids':[r['id'] for r in chosen]})
  result={'arm':name,'status':status,'elapsed_seconds':time.monotonic()-start,'licensed_clauses':len(clauses),'see_subject_derivations_blocked':blocked,'join_receipts':receipts,'all_combinations':sum(len(clauses)**n for n in (2,3,4)),'exact_output_occurrences':len(rows),'unique_normalized_outputs':len({normalize_letters(r['text']) for r in rows}),'mechanically_eligible':sum(r['mechanically_eligible'] for r in rows),'sample':sample,'sampling_manifest':manifest,'outputs':rows,'human_verified_novel_paragraphs':0};results.append(result)
  print(json.dumps({k:v for k,v in result.items() if k not in ('outputs','sample','sampling_manifest','join_receipts')}),flush=True)
 # Frozen exhaustive baseline comparison proves complete residual joining.
 old=json.loads((OUT/'complete-constituent-join-001.json').read_text());assert {r['text'] for r in results[0]['outputs']}=={r['text'] for r in old['outputs']}
 record={'plan':plan,'results':results,'complete_core_output_set_matches_prior_exhaustive':True,'human_review_queue':[],'interpretation':'Removing filler fragments does not preserve palindrome exactness automatically; every recomposed whole stream was rechecked. Subject semantics removes the measured inanimate-sees defect, not all meaning/coherence failures.'}
 # Queue <=5 genuinely different constructions, retain as candidates only.
 promising=[r for r in results[-1]['outputs'] if r['mechanically_eligible'] and len(r['predicate_types'])>=2 and not r['inanimate_sees_clauses']]
 promising.sort(key=lambda r:(r['text'].lower().count('no rider'),-len(r['predicate_types']),r['letters'],r['id']))
 seen=set()
 for r in promising:
  if normalize_letters(r['text']) in seen:continue
  seen.add(normalize_letters(r['text']));record['human_review_queue'].append({**r,'why_borderline':'Complete clauses, exactness and source exclusions passed; animate seeing subjects and verb variation. Cross-clause coherence still needs human review.'})
  if len(record['human_review_queue'])==5:break
 (OUT/'fragment-salvage-recomposition-001.json').write_text(json.dumps(record,indent=2)+'\n');print('review queue',len(record['human_review_queue']))
if __name__=='__main__':run()
