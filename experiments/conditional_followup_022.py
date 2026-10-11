"""Posthoc grammatical endpoint change, inside the original 30-second cap."""
import sys,json,itertools,time,math,random
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.bilateral_seams import Chunk,AllChunkCentersGrammar,OwedSideBilateralGrammar,screen_fixed_endpoints,norm
D=ROOT/'research/block-seams/linked-arguments-022/conditional-followup'
SOURCE='022 posthoc conditional office frame; primary repeats memos explicitly. Weakened condition is a semantic diagnostic only.'

def path(name,texts,roles):
 states=['START']+[name+'-S'+str(i) for i in range(len(texts)-1)]+['END']
 return [Chunk(name+'-'+str(i),t,states[i],states[i+1],roles[i],SOURCE) for i,t in enumerate(texts)]

def run():
 if (D/'summary.json').exists():raise RuntimeError('receipt exists; do not rerun')
 D.mkdir(parents=True,exist_ok=True)
 with (D/'active.lock').open('x') as f:f.write('conditional_followup_022 sole owner\n')
 try:
  plan=dict(status='Explicit posthoc hypothesis change following root/nominal failures; prior artifacts untouched',primary='Some men, if error(s) in the memos is/are ADJ, refine memos.',diagnostic='Some men, if editors VERB OBJECT, define memos. Only OBJECT=memos is an explicit shared-document link.',max_steps=3,max_states=10000,max_outputs=100,seconds_per_search=2,combined_022_cap=30,paid_calls=0,seed=19033)
  (D/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');started=time.monotonic();proposals=[];compiled={}
  for noun,copula,adj in itertools.product(('error','errors'),('is','are'),('minor','major','rare','clear','severe','chronic','persistent','obvious','serious')):
   if (noun=='error')!=(copula=='is'):continue
   name='repair-'+noun+'-'+adj
   cs=path(name,['Some men, if '+noun+' ','in the memos '+copula+' ',adj+', refine memos.'],['main subject, conditional marker, mass/singular or plural error subject','same-document PP and agreeing conditional copula','conditional predicate, clause boundary, main predicate and shared memos object'])
   screen=screen_fixed_endpoints([cs[0]],[cs[-1]])
   text=''.join(c.text for c in cs);t=norm(text);proposals.append(dict(arm='explicit_error_memos',name=name,text=text,tape=t,exact=t==t[::-1],screen=screen,chunks=[asdict(c) for c in cs],explicit_semantic_link=True))
   if screen['admitted']:compiled.setdefault('explicit_error_memos',[]).extend(cs)
  for verb,obj in itertools.product(('read','review','edit','parse','assess','discuss','understand','evaluate','suss'),('memos','notes','reports','forms','plans','menus','recipes','roti','grammar','messages')):
   name='editor-'+verb+'-'+obj;cs=path(name,['Some men, if editors ',verb+' ',obj+', define memos.'],['main subject and conditional editor subject','complete lexical transitive verb in conditional clause','conditional object, clause boundary, main finite predicate and memos object'])
   screen=screen_fixed_endpoints([cs[0]],[cs[-1]])
   text=''.join(c.text for c in cs);t=norm(text);proposals.append(dict(arm='editor_condition',name=name,text=text,tape=t,exact=t==t[::-1],screen=screen,chunks=[asdict(c) for c in cs],explicit_semantic_link=obj=='memos',tentative_lexical_choice=verb=='suss'))
   if screen['admitted']:compiled.setdefault('editor_condition',[]).extend(cs)
  results=[];sample=[];rng=random.Random(plan['seed'])
  for arm in ('explicit_error_memos','editor_condition'):
   cs=compiled.get(arm,[]);expected={r['tape'] for r in proposals if r['arm']==arm and r['exact']}
   for method,cls in [('whole_chunk_centers',AllChunkCentersGrammar),('owed_outside_in',OwedSideBilateralGrammar)]:
    if time.monotonic()-started+json.loads((D.parent/'summary.json').read_text())['elapsed_seconds']>30:raise RuntimeError('combined cap reached')
    if not cs:r=dict(outputs=[],failures=[],receipt=dict(status='skipped_endpoint_mismatch',states_visited=0,failures=0,pending_states=0))
    else:
     g=cls(cs,max_seeds=5000) if cls is AllChunkCentersGrammar else cls(cs);r=g.search(max_steps=3,max_states=10000,max_outputs=100,seconds=2)
     assert r['receipt']['status']=='complete_bounded_depth'
    assert {o['tape'] for o in r['outputs']}==expected
    results.append(dict(arm=arm,method=method,**r));k=math.ceil(.1*len(r['outputs']))
    sample.extend(dict(arm=arm,method=method,**o,review_pending=True,human_label=None) for o in rng.sample(r['outputs'],k))
  summary=dict(status='complete_posthoc_conditional_test',proposals=len(proposals),early_rejected=sum(not r['screen']['admitted'] for r in proposals),explicit_link_exact=sum(r['exact'] and r['explicit_semantic_link'] for r in proposals),weakened_link_exact=sum(r['exact'] and not r['explicit_semantic_link'] for r in proposals),elapsed_seconds=time.monotonic()-started,shared_argument_quality_claim=False)
  for name,value in [('proposals.json',proposals),('engine-results.json',results),('sample-occurrences.json',sample),('summary.json',summary)]:
   (D/name).write_text(json.dumps(value,indent=2)+'\n')
  print(json.dumps(summary));print(json.dumps([r['text'] for r in proposals if r['exact']]))
 finally:(D/'active.lock').unlink()
if __name__=='__main__':run()
