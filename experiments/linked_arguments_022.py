"""Finite two-clause document/coreference hypothesis with fixed-edge screening."""
import sys,json,time,itertools,math,random,hashlib
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.bilateral_seams import Chunk,AllChunkCentersGrammar,OwedSideBilateralGrammar,screen_fixed_endpoints,debt,norm
D=ROOT/'research/block-seams/linked-arguments-022'
ACTORS={'An aide':('reads','rips','writes','sorts','drafts','edits'),'An AI':('designs','defines','derives','delivers','drafts')}
EFFECTS=('guide','inform','inspire','prepare','advise','brief','warn','revile')
DOCS=('notes','memos','letters','reports','papers','briefs','forms','plans')
QUANTS=('the','some','nine','two','many','those');REFS=('the','those','these')
REFERENCE='An aide rips nine memos; some men inspire Diana.'
SOURCE='022 authored finite shared-document narrative; Diana fixed as recipient, no name-variant search.'
PLAN=dict(hypothesis='Agent acts on documents; explicitly coreferential plural documents affect recipient in a causal/concessive second clause.',root_pairs=88,document_heads=8,object_determiners=6,anaphoric_determiners=3,potential_primary_paths=12672,max_steps=3,max_states=10000,max_outputs=100,seconds_per_search=2,aggregate_seconds=30,max_seeds=5000,seed=19032,paid_calls=0,endpoint_hard_gate='immutable letter mismatch or actor conflict only',payoff='rank only',repetition='soft; repeated document noun is a semantic requirement, not vetoed',rubric='Frozen 019 grammar/readability/coherence 0-5, holistic 0-10; model grades only, human label null',sampling='ceil(10% outputs) per method/stratum; preserve duplicate method occurrences')

def chunk(i,text,entry,exit,role,bindings=()):return Chunk(i,text,entry,exit,role,SOURCE,bindings)

def run():
 if (D/'summary.json').exists():raise RuntimeError('completed receipt exists; do not rerun')
 D.mkdir(parents=True,exist_ok=True)
 with (D/'active.lock').open('x') as f:f.write('linked_arguments_022 sole owner\n')
 try:
  (D/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n');started=time.monotonic()
  cache={norm(r['text']) for r in json.loads((ROOT/'data/v3_bank.json').read_text())}
  for o in json.loads((ROOT/'research/block-seams/endpoint-frames-021/summary.json').read_text())['outputs']:cache.add(o['tape'])
  roots=[];frontiers=[];proposals=[];primary_chunks=[]
  for actor,actions in ACTORS.items():
   for action,effect in itertools.product(actions,EFFECTS):
    name=actor.replace(' ','-')+'-'+action+'-'+effect
    left=chunk(name+'-L',actor+' '+action+' ','START','OBJECT_DET','subject and finite transitive predicate')
    tail=chunk(name+'-R',effect+' Diana.','SECOND_VP','END','plural finite predicate and human recipient')
    root=screen_fixed_endpoints([left],[tail]);roots.append(dict(actor=actor,action=action,effect=effect,screen=root,potential_leaves=144))
    if not root['admitted']:continue
    conn=', yet ' if action=='rips' else ', so '
    for doc,q in itertools.product(DOCS,QUANTS):
     binding=(('document',doc),('recipient','Diana'))
     prefix=chunk(name+'-'+doc+'-'+q+'-P',actor+' '+action+' '+q+' ','START','OBJECT_HEAD','subject, finite action, pending plural document NP',binding)
     suffix=chunk(name+'-'+doc+'-'+q+'-S',doc+' '+effect+' Diana.','SECOND_HEAD','END','anaphoric plural document head, finite effect, recipient',binding)
     prior=screen_fixed_endpoints([left],[suffix])
     screen=screen_fixed_endpoints([prefix],[suffix],previous=prior.get('debt') if prior['admitted'] else None,extension_side='left' if prior['admitted'] else None,extension_text=q)
     frontiers.append(dict(actor=actor,action=action,effect=effect,doc=doc,quantifier=q,screen=screen,anaphor_expansions_skipped=3 if not screen['admitted'] else 0))
     for ref in REFS:
      full=actor+' '+action+' '+q+' '+doc+conn+ref+' '+doc+' '+effect+' Diana.'
      tape=norm(full);entry=dict(actor=actor,action=action,effect=effect,document=doc,quantifier=q,anaphor=ref,connector=conn,text=full,tape=tape,exact=tape==tape[::-1],previously_cached=tape in cache,endpoint_admitted=screen['admitted'],semantic_link='Second NP explicitly repeats first document head; sender/action and document/effect have separate argument slots.')
      proposals.append(entry)
      if not screen['admitted']:continue
      case=name+'-'+doc+'-'+q+'-'+ref
      primary_chunks += [chunk(case+'-L',actor+' '+action+' '+q+' ','START',case+'-DOC','subject/finite action/object determiner',binding),chunk(case+'-M',doc+conn+ref+' ',case+'-DOC',case+'-REF','document object, causal/concessive seam, anaphoric determiner',binding),chunk(case+'-R',doc+' '+effect+' Diana.',case+'-REF','END','same plural document as subject, effect, recipient',binding)]
  # Historical two-clause control plus actual causal/concessive marker changes.
  refs=[];reference_chunks=[]
  for label,conn in [('reference','; '),('causal',', so '),('concessive',', yet '),('additive',', and ')]:
   text='An aide rips nine memos'+conn+'some men inspire Diana.';t=norm(text)
   refs.append(dict(label=label,text=text,tape=t,exact=t==t[::-1],explicit_document_coreference=False,provenance='Prior checkpoint reference, not a new or human-approved paragraph'))
   reference_chunks += [chunk(label+'-L','An aide rips nine ','START',label+'-OBJ','subject, action, numeric object determiner'),chunk(label+'-M','memos'+conn+'some men ',label+'-OBJ',label+'-VP','object and second human subject; no explicit document anaphor'),chunk(label+'-R','inspire Diana.',label+'-VP','END','plural finite predicate and recipient')]
  results=[];sample=[];rng=random.Random(PLAN['seed'])
  for arm,cs,expected in [('shared_document',primary_chunks,{r['tape'] for r in proposals if r['exact']}),('historical_reference',reference_chunks,{r['tape'] for r in refs if r['exact']})]:
   for method,cls in [('whole_chunk_centers',AllChunkCentersGrammar),('owed_outside_in',OwedSideBilateralGrammar)]:
    if time.monotonic()-started>30:raise RuntimeError('aggregate budget reached')
    if not cs:r=dict(outputs=[],failures=[],receipt=dict(status='skipped_proved_endpoint_mismatch',states_visited=0,failures=0,pending_states=0))
    else:
     g=cls(cs,max_seeds=5000) if cls is AllChunkCentersGrammar else cls(cs)
     r=g.search(max_steps=3,max_states=10000,max_outputs=100,seconds=2)
     assert r['receipt']['status']=='complete_bounded_depth'
    assert {o['tape'] for o in r['outputs']}==expected
    for o in r['outputs']:assert o['tape']==o['tape'][::-1]
    results.append(dict(arm=arm,method=method,**r))
    k=math.ceil(.1*len(r['outputs']));sample.extend(dict(arm=arm,method=method,**o,human_label=None,review_pending=True) for o in rng.sample(r['outputs'],k))
  summary=dict(status='complete_evaluated_letter_hypothesis',root_pairs=len(roots),root_pairs_admitted=sum(r['screen']['admitted'] for r in roots),potential_grid_paths=12672,root_pruned_virtual_paths=sum(r['potential_leaves'] for r in roots if not r['screen']['admitted']),nominal_frontiers=len(frontiers),nominal_frontiers_admitted=sum(r['screen']['admitted'] for r in frontiers),rendered_primary_proposals=len(proposals),primary_exact=sum(r['exact'] for r in proposals),primary_compiled_chunks=len(primary_chunks),reference_paths=len(refs),reference_exact=sum(r['exact'] for r in refs),reference_letters=len(norm(REFERENCE)),sampled_occurrences=len(sample),elapsed_seconds=time.monotonic()-started,claim='No primary coherent two-clause output; reference exactness is a control, not progress.')
  for name,value in [('root-screens.json',roots),('nominal-frontiers.json',frontiers),('primary-rejected-proposals.json',proposals),('reference-attempts.json',refs),('reference-inventory.json',[asdict(c) for c in reference_chunks]),('engine-results.json',results),('sample-occurrences.json',sample),('summary.json',summary)]:
   (D/name).write_text(json.dumps(value,indent=2)+'\n')
  print(json.dumps(summary))
 finally:(D/'active.lock').unlink()
if __name__=='__main__':run()
