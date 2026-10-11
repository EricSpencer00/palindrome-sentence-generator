"""Request-driven lexical boundary generation with pre-interior viability gate."""
import sys,json,time,math,random
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.phrase_completion import PhraseCompleter,CONCEPTS,RESOURCES
from llm_palindrome.bilateral_seams import Chunk,debt,AllChunkCentersGrammar,OwedSideBilateralGrammar
from llm_palindrome.typed_phrase_compiler import PhraseRule,compile_phrase_rules,repetition_penalty
D=ROOT/'research/block-seams/lexical-boundaries-025'
CONSTRUCTIONS={'negative_advice':('pause_link','cognitive_link'),'direct_deliberation':('bridge','location_link'),'purpose_advice':('location_link','cognitive_link'),'modal_deliberation':('location_link','cognitive_link')}
PLAN={'meaning':'In an already identified room, the recipient should take care and deliberate about responding to the identified question; prohibit haste, directly consider the question, take time, or pause/wait under a modal. Optional location phrases refer to that same room.','max_constructions':4,'completion_requests':128,'phrases_per_request':16,'phrase_words':[2,6],'seed':19036,'aggregate_cpu_seconds':30,'max_chunks':8,'max_states':10000,'max_outputs':100,'max_seeds':5000,'seconds_per_search':7,'gate':'Require a parsed opening/ending pair plus two licensed whole-chunk extensions each paying at least one owed letter. If empty, do not invoke interior search.','repetition':'Finite cost, never veto; payoff guides only; center-out closure requires empty debt.','quality_rubric':'Frozen 019 dimensions 0-5, holistic 0-10, actual human preference labels separately preserved.','training_preferences':'The ten positively labeled examples are preferences/controls, not held-out evaluation.','lexical_scope':'Frozen Brown POS + existing dictionary and morphology, plus explicitly authored finite semantic classes and constructions. No general-English coverage or phrase-attestation claim.','paid_calls':0}

def run():
 if (D/'summary.json').exists():raise RuntimeError('receipt exists; do not rerun')
 D.mkdir(parents=True,exist_ok=True)
 with (D/'active.lock').open('x') as f:f.write('025 sole research owner\n')
 try:
  started=time.process_time();wall=time.monotonic();g=PhraseCompleter();(D/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n')
  b={'observer':'geese1','patient':'God1','subject_number':'plural'};args={'predicate':'perceive','patient_surface':'God','construction':'question_do','bindings':b,'required_bindings':dict(b)}
  control=g.complete_phrase(debt('Do geese',''),'right','perception_vp',args);assert len(control['phrases'])==1
  cp=control['phrases'][0];assert cp['text']=='see God' and cp['paid_letters']==6 and cp['debt']=={'side':'left','letters':'e'}
  bad=[]
  for k,v in [('patient','other1'),('subject_number','singular')]:
   a={**args,'bindings':{**b,k:v}};r=g.complete_phrase(debt('Do geese',''),'right','perception_vp',a);assert not r['phrases'];bad.append(r)
  (D/'generator-control.json').write_text(json.dumps({'control_only':True,'new_creative_result':False,'committed':'Do geese','full_vp_input_supplied':False,'result':control,'conflicting_arguments':bad},indent=2)+'\n')
  roots=[];viable=[];bound={'responder':'reader1','question':'question1','location':'room1','subject_person':'second','subject_number':'singular'}
  for c,links in CONSTRUCTIONS.items():
   a={'construction':c,'patient_surface':'question','location_surface':'room','bindings':bound,'required_bindings':dict(bound)}
   openings=g.complete_phrase({'side':'none','letters':''},'left','opening',a)
   for opening in openings['phrases']:
    ends=g.complete_phrase(opening['debt'],'right','ending',a)
    for ending in ends['phrases']:
     state={'left':opening['text']+('. ' if c=='negative_advice' else ' '),'right':ending['text']+'.','debt':debt(opening['text'],ending['text']),'opening':opening,'ending':ending,'extensions':[]}
     assert state['debt']['compatible'];frontier=[state]
     # For the viable negative case, choose the right cognitive link first.
     # Each extension is a parsed 2-6-word phrase, never a lexical half-word.
     schedule=[('right',links[-1]),('left',links[0])]
     for direction,role in schedule:
      nxt=[]
      for s in frontier:
       if s['debt']['side']==direction:
        roots.append({'construction':c,'stage':role,'status':'wrong_debt_owner_for_required_interface','state':s});continue
       r=g.complete_phrase(s['debt'],direction,role,a)
       if not r['phrases']:roots.append({'construction':c,'stage':role,'status':'no_role_licensed_boundary_completion','state':s,'failures':r['failures']})
       for p in r['phrases']:
        if p['paid_letters']<1:roots.append({'construction':c,'stage':role,'status':'extension_not_debt_paying','phrase':p});continue
        left=s['left']+p['text']+' ' if direction=='left' else s['left'];right=p['text']+' '+s['right'] if direction=='right' else s['right'];d=debt(left,right);assert d['compatible']
        nxt.append({**s,'left':left,'right':right,'debt':d,'extensions':s['extensions']+[{'direction':direction,'role':role,'phrase':p}]})
      frontier=nxt
     viable.extend({'construction':c,**s} for s in frontier)
    if not ends['phrases']:roots.append({'construction':c,'stage':'ending','status':'no_parsed_endpoint_pair','opening':opening,'failures':ends['failures']})
  (D/'completion-requests.json').write_text(json.dumps(g.calls,indent=2)+'\n');(D/'endpoint-gate.json').write_text(json.dumps({'viable':viable,'failures':roots},indent=2)+'\n')
  interior=[]
  if viable:
   chunks=[]
   for c in sorted({s['construction'] for s in viable}):
    role_order=('opening',*CONSTRUCTIONS[c],'ending');states=('START','AFTER_OPENING','AFTER_LINK1','AFTER_LINK2','END');rules=[]
    for j,role in enumerate(role_order):
     for call in g.calls:
      if call['role']!=role or call['arguments'].get('construction')!=c:continue
      for phrase in call['phrases']:
       text=phrase['text']+('. ' if role=='opening' and c=='negative_advice' else ('.' if role=='ending' else ' '))
       rules.append(PhraseRule(states[j],states[j+1],text,role,tuple(sorted(phrase['bindings'].items())),'025 independently parsed lexical completion; derivation '+phrase['derivation']))
    chunks.extend(compile_phrase_rules('025-'+c,rules,set(zip(states,states[1:]))))
   (D/'interior-inventory.json').write_text(json.dumps([asdict(c) for c in chunks],indent=2)+'\n')
   for method,cls in [('whole_chunk_centers',AllChunkCentersGrammar),('owed_outside_in',OwedSideBilateralGrammar)]:
    remaining=PLAN['aggregate_cpu_seconds']-(time.process_time()-started)
    if remaining<=0:raise RuntimeError('aggregate CPU budget exhausted before search')
    engine=cls(chunks,max_seeds=PLAN['max_seeds']) if cls is AllChunkCentersGrammar else cls(chunks)
    result=engine.search(max_steps=PLAN['max_chunks'],max_states=PLAN['max_states'],max_outputs=PLAN['max_outputs'],seconds=min(PLAN['seconds_per_search'],remaining));result['method']=method
    for output in result['outputs']:
     tape=''.join(ch.lower() for ch in output['text'] if ch.isascii() and ch.isalpha());assert tape==tape[::-1] and tape==output['tape']
     output['finite_repetition_cost']=repetition_penalty(output['text']);output['human_label']=None;output['owner_quality_review_pending']=True
    interior.append(result)
   (D/'interior-results.json').write_text(json.dumps(interior,indent=2)+'\n')
  rng=random.Random(PLAN['seed']);samples=[]
  for result in interior:
   for output in rng.sample(result['outputs'],math.ceil(.1*len(result['outputs']))):samples.append({'method':result['method'],**output})
  (D/'sample-occurrences.json').write_text(json.dumps(samples,indent=2)+'\n')
  assert time.process_time()-started<PLAN['aggregate_cpu_seconds']
  summary={'status':'complete_finite_empty_endpoint_gate' if not viable else ('complete_bounded_interior' if all(r['receipt']['status']=='complete_bounded_depth' for r in interior) else 'incomplete_interior'),'requests':len(g.calls),'request_cap':128,'phrase_cap':16,'constructions':len(CONSTRUCTIONS),'opening_phrases':sum(len(r['phrases']) for r in g.calls if r['role']=='opening'),'parsed_endpoint_pairs':sum(len(r['phrases']) for r in g.calls if r['role']=='ending'),'two_extension_survivors':len(viable),'interior_search_invocations':len(interior),'creative_outputs':sum(len(r['outputs']) for r in interior),'generated_quality_reviews':[],'sample_occurrences':len(samples),'sample_percentage':None,'control':{'paid_letters':6,'residual':'e','new_creative_result':False},'cpu_seconds':time.process_time()-started,'wall_seconds':time.monotonic()-wall,'resource_hashes':g.resource_hashes,'completion_statuses':sorted(set(r['receipt']['status'] for r in g.calls))}
  (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 finally:(D/'active.lock').unlink(missing_ok=True)
if __name__=='__main__':run()
