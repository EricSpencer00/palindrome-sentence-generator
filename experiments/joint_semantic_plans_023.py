"""Meaning-first joint surface choice: editing causes and drawing request/reply.
No new names or unlinked objects are invented to repair letter debt.
"""
import sys,json,time,itertools,math,random
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.bilateral_seams import Chunk,AllChunkCentersGrammar,OwedSideBilateralGrammar,screen_fixed_endpoints,norm
D=ROOT/'research/block-seams/joint-semantic-plans-023'
SOURCE='023 authored semantic plans before surface compatibility; no attestation or world-originality claim.'
EDITING=(('An editor','corrects','notes','readers','understand'),('Some editors','revise','reports','readers','follow'),('We','check','memos','the team','use'),('Some men','review','memos','the team','refine'),('Editors','resolve errors in','forms','clerks','process'),('The reviewers','mark errors in','papers','writers','revise'),('The writer','clarifies','plans','the reader','follow'),('A team','revises','briefs','the audience','understand'))
VERBS={'draw':'drawing','sketch':'sketching','outline':'outlining','map':'mapping','diagram':'diagramming','design':'designing'}
MODS=('', 'mock','new','small','large','empty','clean','private')
TYPES=('', 'coma','trauma','medical','surgical','cardiac','recovery','maternity')
REFS=('indefinite_full','definite_full','demonstrative_full','omit_modifier','ward_only','pronoun')
PLAN=dict(semantic_plans=['Errors are corrected in a document, enabling a recipient to use that same document.','A person/group receives a request to depict a fictional hospital ward, then acknowledges drawing that same ward; surface NP aliases preserve one drawing/topic identity.'],editing_plan_count=8,editing_orders=3,drawing_command_verbs=6,drawing_response_verbs=6,drawing_aspects=2,drawing_response_actors=2,document_modifier_choices=8,document_type_choices=8,anaphoric_forms=6,virtual_drawing_paths=55296,max_steps=4,max_states=10000,max_outputs=100,seconds_per_search=2,aggregate_seconds=30,max_seeds=5000,seed=19034,paid_calls=0,grammar='Imperative addressee is response actor; first-person agreement; full NP and definite/demonstrative/pronominal references preserve document binding.',repetition='Soft token/phrase penalty, never a hard ban; repeated confirmation is labeled.',rubric='Frozen 019 grammar/readability/coherence 0-5; holistic overall 0-10; model judgments only.',provenance='Authored bounded language, not an unbiased natural-language yield estimate; drawing seed draw a ward is a known component.')

def chunks(name,texts,roles,doc,actor):
 states=['START']+[name+'-G'+str(i) for i in range(len(texts)-1)]+['END'];binding=(('document',doc),('performer',actor))
 return [Chunk(name+'-'+str(i),t,states[i],states[i+1],roles[i],SOURCE,binding) for i,t in enumerate(texts)]

def np(mod,care,ref='indefinite_full'):
 words=[w for w in (mod,care,'ward') if w];body=' '.join(words)
 if ref=='pronoun':return 'it'
 if ref=='ward_only':return 'the ward'
 if ref=='omit_modifier':return 'the '+(' '.join(w for w in (care,'ward') if w))
 if ref=='definite_full':return 'the '+body
 if ref=='demonstrative_full':return 'that '+body
 return ('an ' if body[0] in 'aeiou' else 'a ')+body

def run():
 if (D/'summary.json').exists():raise RuntimeError('receipt exists; do not rerun')
 D.mkdir(parents=True,exist_ok=True)
 with (D/'active.lock').open('x') as f:f.write('joint_semantic_plans_023 sole owner\n')
 try:
  (D/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n');started=time.monotonic();records=[];inventories={'editing':[],'drawing_dialogue':[]};roots=[]
  cache={norm(r['text']) for r in json.loads((ROOT/'data/v3_bank.json').read_text())}
  # Explicitly preserve the rejected 022 candidate's status in comparisons.
  rejected=norm('Some men, if editors suss roti, define memos.')
  for r in json.loads((ROOT/'research/block-seams/endpoint-frames-021/summary.json').read_text())['outputs']:cache.add(r['tape'])
  for i,(actor,action,doc,reader,effect) in enumerate(EDITING):
   first=f'{actor} {action} the {doc}';last=f'{reader} can {effect} the {doc}.'
   for order in ('causal','conditional','two_sentences'):
    lead='If '+first[0].lower()+first[1:]+', ' if order=='conditional' else first+(', so ' if order=='causal' else '. Therefore, ')
    name=f'editing-{i}-{order}';cs=chunks(name,[lead,last],['independent actor/document action or condition','effect on explicitly same document'],doc,actor.lower());screen=screen_fixed_endpoints([cs[0]],[cs[-1]])
    text=lead+last;tape=norm(text);row=dict(arm='editing',name=name,text=text,tape=tape,exact=tape==tape[::-1],screen=screen,document=doc,semantic_relation='Same document, action and enabling effect, fixed before letter checks.',previously_cached=tape in cache,rejected_022_reused=tape==rejected)
    records.append(row)
    if screen['admitted']:inventories['editing'].extend(cs)
  # Screen command/document edges before generating response/NP choices.
  # A drawing can depict a ward; no clinical facility recommendation is implied.
  for command,head in itertools.product(VERBS,('ward','it')):
   cs=chunks('root-'+command+'-'+head,[command.capitalize()+' ',head],['requested depiction action','same depicted object or pronoun'], 'ward_plan','response_actor')
   gate=screen_fixed_endpoints([cs[0]],[cs[-1]]);roots.append(dict(command=command,final_head=head,screen=gate))
   if not gate['admitted']:continue
   refs=REFS[:-1] if head=='ward' else REFS[-1:]
   for mod,care,ref,actor,response,aspect in itertools.product(MODS,TYPES,refs,('I','we'),VERBS,('present','progressive')):
    first_np=np(mod,care);second_np=np(mod,care,ref)
    verb=response if aspect=='present' else ('am ' if actor=='I' else 'are ')+VERBS[response]
    reply=actor.capitalize()+' '+verb+' '
    identity='ward-plan-'+(mod or 'plain')+'-'+(care or 'general')
    name=f'{identity}-{ref}-{actor}-{response}-{aspect}'
    cs=chunks(name,[command.capitalize()+' ',first_np+'. ',reply,second_np+'.'],['imperative depiction verb','introduced drawing/topic NP; command completes','bound addressee acknowledges same action; first-person agreement','anaphoric NP of same drawing/topic'],identity,actor)
    gate=screen_fixed_endpoints(cs[:2],cs[2:]);text=''.join(c.text for c in cs);tape=norm(text)
    row=dict(arm='drawing_dialogue',name=name,text=text,tape=tape,exact=tape==tape[::-1],screen=gate,document=identity,command_verb=command,response_verb=response,aspect=aspect,response_actor=actor,reference=ref,modifier=mod,care_type=care,semantic_relation='Request followed by acknowledgment; same performer and depicted ward, no extra event invented.',previously_cached=tape in cache,rejected_022_reused=tape==rejected)
    records.append(row)
    if gate['admitted']:inventories['drawing_dialogue'].extend(cs)
  for arm,cs in inventories.items():
   if len(cs)>512:raise RuntimeError('finite compilation bound exceeded; preserve before expanding')
  results=[];sample=[];rng=random.Random(PLAN['seed'])
  for arm,cs in inventories.items():
   expected={r['tape'] for r in records if r['arm']==arm and r['exact']}
   for method,cls in [('whole_chunk_centers',AllChunkCentersGrammar),('owed_outside_in',OwedSideBilateralGrammar)]:
    if time.monotonic()-started>30:raise RuntimeError('aggregate wall budget reached')
    if not cs:r=dict(outputs=[],failures=[],receipt=dict(status='skipped_proved_endpoint_mismatch',states_visited=0,failures=0,pending_states=0))
    else:
     g=cls(cs,max_seeds=5000) if cls is AllChunkCentersGrammar else cls(cs);r=g.search(max_steps=4,max_states=10000,max_outputs=100,seconds=2)
     assert r['receipt']['status']=='complete_bounded_depth'
    assert {o['tape'] for o in r['outputs']}==expected
    for o in r['outputs']:assert o['tape']==o['tape'][::-1]
    results.append(dict(arm=arm,method=method,**r));k=math.ceil(.1*len(r['outputs']));sample.extend(dict(arm=arm,method=method,**o,review_pending=True,human_label=None) for o in rng.sample(r['outputs'],k))
  for name,value in [('endpoint-roots.json',roots),('proposals.json',records),('inventory.json',{arm:[asdict(c) for c in cs] for arm,cs in inventories.items()}),('engine-results.json',results),('sample-occurrences.json',sample)]:
   (D/name).write_text(json.dumps(value,indent=2)+'\n')
  unique={o['tape']:o for r in results for o in r['outputs']}
  summary=dict(status='complete_bounded_joint_plans',root_checks=len(roots),root_admitted=sum(r['screen']['admitted'] for r in roots),virtual_drawing_paths=55296,actual_proposals=len(records),editing_proposals=sum(r['arm']=='editing' for r in records),drawing_proposals=sum(r['arm']=='drawing_dialogue' for r in records),early_rejected=sum(not r['screen']['admitted'] for r in records),compiled_chunks={arm:len(cs) for arm,cs in inventories.items()},unique_outputs=len(unique),outputs=list(unique.values()),sampled_occurrences=len(sample),elapsed_seconds=time.monotonic()-started,quality_review_pending=True,claimed_new_best=False)
  (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='outputs'}));print(json.dumps([o['text'] for o in unique.values()]))
 finally:(D/'active.lock').unlink()
if __name__=='__main__':run()
