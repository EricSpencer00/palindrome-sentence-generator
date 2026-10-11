"""One finite meaning plan compiled from independent phrases, never candidate texts.
Complete donors provide provenance only; no complete-text screening precedes compilation.
"""
import sys,json,time,math,random
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.typed_phrase_compiler import PhraseRule,compile_phrase_rules,repetition_penalty
from llm_palindrome.bilateral_seams import Chunk,AllChunkCentersGrammar,OwedSideBilateralGrammar,screen_fixed_endpoints,norm
D=ROOT/'research/block-seams/typed-composition-024'
PLAN={'meaning':'Tell the question recipient to avoid a hasty response, then pause and deliberate about whether or how to acknowledge or answer that same question.','entities':{'responder':'reader1','question':'question1'},'grammar':'Two second-person imperatives; the second controls a purpose infinitive taking an interrogative infinitival complement. Nod, answer and respond are licensed response actions, not identical events.','rules_cap':32,'chunk_cap':8,'state_cap':10000,'output_cap':100,'seed_cap':5000,'search_seconds':7,'aggregate_cpu_seconds':30,'sample_seed':19035,'paid_calls':0,'rule_licensing':'Owner-authored English phrase parses, not external lexical attestation; no corpus or world-originality claims.','quality_rubric':'Frozen 019 grammar/readability/coherence 0-5 and holistic overall 0-10; human labels null.','repetition':'Finite token+0.5 repeated 2/3/4-gram excess, never an admission rule. Payoff guidance only.'}
STATES=('START','NEGATIVE_BASE_VP','SECOND_IMPERATIVE','PURPOSE_INFINITIVE','INTERROGATIVE_COMPLEMENT','RESPONSE_INFINITIVE','END')
SURFACES=(('Do not ',),('rush. ','answer immediately. ','respond hastily. ','speak yet. '),('Take a moment ','Pause ','Wait '),('to consider ','to decide ','to think about '),('whether ','how '),('to nod.','to answer.','to respond.'))
ROLES=('negative imperative auxiliary; base-form verb obligation','base-form controlled action completes first imperative','second imperative licenses purpose infinitive','purpose cognitive predicate requires interrogative complement','interrogative introduces controlled response infinitive','response infinitive closes same responder/question plan')
DONORS=((0,0,0,0,0,0),(0,1,1,1,1,1),(0,2,2,2,0,2),(0,3,1,0,1,0))
BINDINGS=(('responder','reader1'),('question','question1'),('imperative_person','second'),('response_controller','reader1'),('complement_type','interrogative_infinitive'))

def inputs():
 donors=[]
 for i,choice in enumerate(DONORS):
  donor=[]
  for j,k in enumerate(choice):
   donor.append(PhraseRule(STATES[j],STATES[j+1],SURFACES[j][k],ROLES[j],BINDINGS,f'024 donor {i}; independently licensed {ROLES[j]}; semantic plan fixed before search'))
  donors.append(donor)
 return donors

def run():
 if (D/'summary.json').exists():raise RuntimeError('receipt exists; do not rerun')
 D.mkdir(parents=True,exist_ok=True)
 with (D/'active.lock').open('x') as f:f.write('024 sole research owner\n')
 try:
  (D/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n');start=time.process_time();wall=time.monotonic();donors=inputs();flat=[r for d in donors for r in d]
  interfaces=set(zip(STATES,STATES[1:]));shared=compile_phrase_rules('deliberate-response',flat,interfaces);assert len(shared)<=PLAN['rules_cap']
  private=[]
  for i,donor in enumerate(donors):
   states=['START']+[f'private{i}-{j}' for j in range(5)]+['END']
   private.extend(Chunk(f'donor{i}-{j}',r.text,states[j],states[j+1],r.role,r.source,r.bindings) for j,r in enumerate(donor))
  # Compare unique licensed phrase content/features, ignoring donor path labels.
  key=lambda r:(r.text,r.role,tuple(r.bindings))
  assert {key(r) for r in private}=={key(r) for r in shared}
  donor_texts=[''.join(r.text for r in d) for d in donors];assert all(norm(t)!=norm(t)[::-1] for t in donor_texts)
  for n,value in [('donors.json',[{'id':i,'text':t,'exact':False,'rules':[asdict(r) for r in d]} for i,(d,t) in enumerate(zip(donors,donor_texts))]),('phrase-rules.json',[asdict(r) for r in flat]),('inventories.json',{'private':[asdict(r) for r in private],'compositional':[asdict(r) for r in shared]})]:
   (D/n).write_text(json.dumps(value,indent=2)+'\n')
  results=[]
  for arm,cs in [('donor_private',private),('typed_compositional',shared)]:
   for method,cls in [('whole_chunk_centers',AllChunkCentersGrammar),('owed_outside_in',OwedSideBilateralGrammar)]:
    remaining=PLAN['aggregate_cpu_seconds']-(time.process_time()-start)
    if remaining<=0:raise RuntimeError('aggregate CPU budget exhausted')
    g=cls(cs,max_seeds=PLAN['seed_cap']) if cls is AllChunkCentersGrammar else cls(cs)
    res=g.search(max_steps=PLAN['chunk_cap'],max_states=PLAN['state_cap'],max_outputs=PLAN['output_cap'],seconds=min(PLAN['search_seconds'],remaining));res.update(arm=arm,method=method)
    for o in res['outputs']:
     tape=norm(o['text']);assert tape==tape[::-1];o.update(absent_from_donor_texts=o['text'] not in donor_texts,finite_repetition_penalty=repetition_penalty(o['text']),human_label=None)
    results.append(res)
  (D/'engine-results.json').write_text(json.dumps(results,indent=2)+'\n')
  # Post-search proof covers mandatory prefix/suffix interfaces, not complete texts.
  certificates=[]
  for response in [c for c in shared if c.exit=='END']:
   root=screen_fixed_endpoints([next(c for c in shared if c.entry=='START')],[response]);certificates.append({'phase':'response_root','tail':response.text,'screen':root})
   if not root['admitted']:continue
   for action in [c for c in shared if c.entry.endswith(':NEGATIVE_BASE_VP')]:
    for ref in [c for c in shared if c.exit.endswith(':RESPONSE_INFINITIVE')]:
     gate=screen_fixed_endpoints([next(c for c in shared if c.entry=='START'),action],[ref,response]);certificates.append({'phase':'mandatory_outer_interfaces','action':action.text,'reference':ref.text,'screen':gate});assert not gate['admitted']
  (D/'endpoint-certificates.json').write_text(json.dumps(certificates,indent=2)+'\n')
  complete=all(r['receipt']['status']=='complete_bounded_depth' for r in results)
  if complete:assert all(not r['outputs'] for r in results)
  samples=[];rng=random.Random(PLAN['sample_seed'])
  for r in results:
   for o in rng.sample(r['outputs'],math.ceil(.1*len(r['outputs']))):samples.append({'arm':r['arm'],'method':r['method'],**o})
  (D/'sample-occurrences.json').write_text(json.dumps(samples,indent=2)+'\n')
  # DAG path count, without rendering the compositional complete language.
  count={'START':1}
  for state in STATES[:-1]:
   entry=state if state=='START' else 'deliberate-response:'+state
   for c in shared:
    if c.entry==entry:count[c.exit]=count.get(c.exit,0)+count.get(entry,0)
  summary={'status':'complete_finite_negative' if complete else 'incomplete','licensed_unique_rules':len(shared),'private_chunks':len(private),'donor_paths':len(donors),'compositional_paths':count['END'],'same_unique_phrase_inventory':True,'presearch_complete_candidates_enumerated':False,'nonpalindromic_donors_retained':True,'creative_output_occurrences':sum(len(r['outputs']) for r in results),'unique_creative_outputs':0 if complete else None,'required_new_output_gate_passed':False,'sampled_occurrences':len(samples),'cpu_seconds':time.process_time()-start,'wall_seconds':time.monotonic()-wall,'regression_known_control_only':True,'new_quality_claim':False,'engine_receipts':[{k:r[k] for k in ('arm','method','receipt')} for r in results]}
  (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 finally:(D/'active.lock').unlink(missing_ok=True)
if __name__=='__main__':run()
