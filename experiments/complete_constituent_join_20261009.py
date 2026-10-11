"""Complete finite composition of existing source-derived constituents; no target whitelist."""
import hashlib,itertools,json,math,time
from collections import defaultdict
from pathlib import Path
from experiments.derived_paragraph_controls_20261009 import UNITS,CANDIDATES
from experiments.block_seam_comparison_20261009 import render_paragraph
from llm_palindrome.typed_constituents import TypedGrammar,words
from llm_palindrome.admission import normalize_letters
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/block-seams'
def run():
 target=OUT/'complete-constituent-join-001.json'
 if target.exists():raise RuntimeError('immutable run exists')
 inv=json.loads((OUT/'licensed-control-inventory-v2-001.json').read_text());components=set(UNITS)
 assert components<=set(inv['blocks'])
 g=TypedGrammar(set(itertools.chain.from_iterable(words(t) for t in components)))
 clauses=set()
 for slots,label in g.paths:
  choices=[tuple(t for t in slot if ' '.join(t) in components) for slot in slots]
  if all(choices):
   clauses.update(tuple(itertools.chain.from_iterable(parts)) for parts in itertools.product(*choices))
 clauses=sorted(clauses);normalized={c:normalize_letters(' '.join(c)) for c in clauses}
 plan={'components':sorted(components),'component_origin':'Existing v2 provenance-derived inventory; small source-derived grammar slice. Supplied success texts are regressions only, not a target whitelist.','clauses':len(clauses),'sentence_counts':[2,3,4],'band':[60,119],'complete_finite_language':True,'resource_bound':'one CPU worker, <=10 seconds, no model/paid calls','methods':['exhaustive clause enumeration','exact residual half-join'],'quality_gates':'Exact global palindrome; 2-4 distinct licensed clauses; known/supplied exclusions; coherence still requires separate review.','generalization_limit':'Grammar/lexicon deliberately small and derived from known examples; output novelty or human readability is not implied.'}
 (OUT/'complete-constituent-join-001-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 start=time.perf_counter();baseline=set();evaluated=0
 for n in (2,3,4):
  for sequence in itertools.product(clauses,repeat=n):
   evaluated+=1;t=''.join(normalized[c] for c in sequence)
   if 60<=len(t)<=119 and t==t[::-1]:baseline.add(sequence)
 baseline_seconds=time.perf_counter()-start
 start=time.perf_counter();joined=set();indexed=0;probes=0
 for n in (2,3,4):
  leftn=n//2;rightn=n-leftn;byreverse=defaultdict(list)
  for right in itertools.product(clauses,repeat=rightn):
   indexed+=1;byreverse[''.join(normalized[c] for c in right)[::-1]].append(right)
  for left in itertools.product(clauses,repeat=leftn):
   probes+=1;t=''.join(normalized[c] for c in left)
   for right in byreverse.get(t,[]):
    sequence=left+right;whole=t+''.join(normalized[c] for c in right)
    if 60<=len(whole)<=119:joined.add(sequence)
 join_seconds=time.perf_counter()-start
 # Odd clause counts can place the character midpoint inside a clause, so a
 # full-tape equality half-join is incomplete when halves differ in length.
 # Recover completeness using the general exact stream check over all odd tuples.
 start=time.perf_counter();odd_checks=0
 for sequence in itertools.product(clauses,repeat=3):
  odd_checks+=1;t=''.join(normalized[c] for c in sequence)
  if 60<=len(t)<=119 and t==t[::-1]:joined.add(sequence)
 join_seconds+=time.perf_counter()-start
 # Even clause counts can also have unequal half lengths. A conservative
 # fallback checks all unequal-length halves; no pruning claim is overstated.
 start=time.perf_counter();fallback=0
 for n in (2,4):
  for sequence in itertools.product(clauses,repeat=n):
   k=n//2;l=''.join(normalized[c] for c in sequence[:k]);r=''.join(normalized[c] for c in sequence[k:])
   if len(l)==len(r):continue
   fallback+=1;t=l+r
   if 60<=len(t)<=119 and t==t[::-1]:joined.add(sequence)
 join_seconds+=time.perf_counter()-start
 assert baseline==joined
 rows=[]
 for seq in sorted(baseline):
  rendered=render_paragraph(tuple((c,g.complete(c)) for c in seq));t=normalize_letters(rendered);assert t==t[::-1];assert g.text_paragraph(rendered,4)
  rows.append({'id':'join-'+hashlib.sha256(rendered.encode()).hexdigest()[:16],'text':rendered,'letters':len(t),'exact':True,'clause_count':len(seq),'distinct_clauses':len(set(seq))==len(seq),'known_or_supplied_exclusion':t in inv['excluded_normalized'],'known_example_component_derivation':True,'mechanically_eligible':len(set(seq))==len(seq) and t not in inv['excluded_normalized'],'assistant_coherence':'Unreviewed; repeated sees predicates and no explicit causal/discourse relation.','human_score':None})
 strata=defaultdict(list)
 for row in rows:strata[row['clause_count']].append(row)
 samples=[];manifest=[]
 for k,rr in sorted(strata.items()):
  count=math.ceil(len(rr)/10);selected=sorted(rr,key=lambda r:hashlib.sha256(('complete-join-sample-001|'+r['id']).encode()).hexdigest())[:count];samples.extend(selected);manifest.append({'approach':'complete_constituent_join','clause_count':k,'outputs':len(rr),'sample_count':count,'seed':'complete-join-sample-001','ids':[r['id'] for r in selected]})
 record={'plan':plan,'baseline':{'complete_combinations':evaluated,'seconds':baseline_seconds},'half_join':{'indexed':indexed,'probes':probes,'odd_checks':odd_checks,'unequal_half_fallback_checks':fallback,'seconds':join_seconds},'output_sets_equal':True,'outputs':rows,'stratified_sample':samples,'sampling_manifest':manifest,'novel_human_verified_paragraphs':0,'interpretation':'Complete finite construction emits candidates. Equal-half indexing alone is not a sound general seam prune; the conservative fallbacks preserve completeness. No speedup or quality claim without the recorded timings and separate review.'}
 target.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:record[k] for k in ('baseline','half_join','output_sets_equal','sampling_manifest')},indent=2));print('outputs',len(rows),'mechanically eligible',sum(r['mechanically_eligible'] for r in rows))
if __name__=='__main__':run()
