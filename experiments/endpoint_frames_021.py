"""Fixed-edge screening, cached 020 replay, and changed non-sees frames."""
import sys,json,time,math,random
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.bilateral_seams import Chunk,AllChunkCentersGrammar,OwedSideBilateralGrammar,screen_fixed_endpoints,debt,norm
D=ROOT/'research/block-seams/endpoint-frames-021'
PLAN=dict(seed=19031,max_steps=4,max_states=10000,max_outputs=100,seconds_per_search=2,aggregate_seconds=20,max_seeds=5000,paid_calls=0,methods=['whole_chunk_centers','owed_outside_in'],changed_frames=['acknowledgment_copular_np','negative_imperative_adverb','negative_subject_emphatic_vp'],endpoint_rule='Only incompatible immutable prefix/suffix or actor conflict is a hard rejection. Compatible nonpalindromic debt remains admitted. Payoff/repetition rank only.',rubric='Frozen 019: grammar/readability/coherence 0-5, holistic 0-10; model review, no inferred human labels',sampling='ceil(10% outputs) per method/arm; duplicate method occurrences retained')
SRC='Finite changed grammatical frame 021; no originality/attestation claim.'

def make_path(name,texts,roles):
 states=['START']+[name+'-slot-'+str(i) for i in range(len(texts)-1)]+['END']
 return [Chunk(name+'-'+str(i),t,states[i],states[i+1],roles[i],SRC) for i,t in enumerate(texts)]

def frame_paths():
 out=[];heads=('war','car','radar','motor','computer','printer','heater');mods=('', 'dark','small','new','old','clear','quiet','wooden','red')
 for head in heads:
  for mod in mods:
   texts=['No, sir, a ']+([mod+' '] if mod else [])+[head+' is on.']
   roles=['acknowledgment, vocative, subject determiner']+(['single whole adjective in subject NP'] if mod else [])+['subject noun head and finite copular predicate']
   out.append(('acknowledgment_copular_np',f'copula-{head}-{mod or "bare"}',make_path(f'copula-{head}-{mod or "bare"}',texts,roles)))
 for adv in ('','ever','never','once','calmly','quietly','just','at night','deliberately'):
  name='imperative-'+(adv.replace(' ','-') or 'bare')
  out.append(('negative_imperative_adverb',name,make_path(name,["Don't ",(adv+' ' if adv else '')+'nod.'],['negative imperative auxiliary','adverbial and base verb with implicit second-person subject'])))
 for adv in ('','ever','never','daily','deeply','directly','duly','once','briefly'):
  name='emphatic-'+(adv or 'bare')
  out.append(('negative_subject_emphatic_vp',name,make_path(name,['No evil ',(adv+' ' if adv else '')+'did live on.'],['negative abstract singular NP','adverbial and emphatic finite VP; metaphorical live on'])))
 return out

def run_search(cs,method):
 if not cs:return dict(outputs=[],failures=[],receipt=dict(status='skipped_no_compatible_endpoints',states_visited=0,failures=0,pending_states=0))
 g=AllChunkCentersGrammar(cs,max_seeds=5000) if method=='whole_chunk_centers' else OwedSideBilateralGrammar(cs)
 return g.search(max_steps=4,max_states=10000,max_outputs=100,seconds=2)

def run():
 if (D/'summary.json').exists():raise RuntimeError('completed receipt exists; do not rerun')
 D.mkdir(parents=True,exist_ok=True)
 with (D/'active.lock').open('x') as f:f.write('endpoint_frames_021 sole owner\n')
 try:
  (D/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n');start=time.monotonic()
  cached=json.loads((ROOT/'research/block-seams/continuation-020/raw-engine-results.json').read_text());cache_records=[];replays=[]
  for arm in ('geese_compound','geese_ordinary','panic_nonsees'):
   attempts=json.loads((ROOT/f'research/block-seams/continuation-020/{arm}-attempts.json').read_text());cs=[];retained=[]
   for i,a in enumerate(attempts):
    words=a['text'].split();split=4 if arm=='panic_nonsees' else 5
    path=make_path(f'{arm}-{i}',[' '.join(words[:split])+' ',' '.join(words[split:])],['fixed prior licensed phrase prefix','fixed prior licensed phrase suffix'])
    screen=screen_fixed_endpoints([path[0]],[path[-1]])
    cache_records.append(dict(arm=arm,text=a['text'],exact=a['exact'],screen=screen))
    if screen['admitted']:cs.extend(path);retained.append(a)
   expected={a['tape'] for a in attempts if a['exact']}
   for method in PLAN['methods']:
    result=run_search(cs,method);assert {o['tape'] for o in result['outputs']}==expected
    assert result['receipt']['status'] in ('complete_bounded_depth','skipped_no_compatible_endpoints')
    old=next(r for r in cached if r['arm']==arm and r['method']==method)
    replays.append(dict(arm=arm,method=method,paths=len(attempts),early_rejected=len(attempts)-len(retained),retained=len(retained),prior_receipt=old['receipt'],**result))
  (D/'cached-020-screen-records.json').write_text(json.dumps(cache_records,indent=2)+'\n')
  (D/'cached-020-replay.json').write_text(json.dumps(replays,indent=2)+'\n')
  known={norm(e['text']) for e in json.loads((ROOT/'data/v3_bank.json').read_text())}
  for r in cached:known.update(o['tape'] for o in r['outputs'])
  for r in json.loads((ROOT/'research/block-seams/center-coverage-018/raw-results.json').read_text()):known.update(o['tape'] for o in r['outputs'])
  paths=frame_paths();proposals=[];newruns=[];sample=[];rng=random.Random(PLAN['seed'])
  for arm in PLAN['changed_frames']:
   selected=[];all_arm=[]
   for a,name,path in paths:
    if a!=arm:continue
    text=''.join(c.text for c in path);tape=norm(text)
    previous=debt(path[0].text,' is on.') if arm=='acknowledgment_copular_np' else None
    head=path[-1].text.split()[0]
    screen=screen_fixed_endpoints([path[0]],[path[-1]],previous=previous,extension_side='right' if previous else None,extension_text=head)
    row=dict(arm=arm,name=name,text=text,tape=tape,exact=tape==tape[::-1],previously_cached=tape in known,screen=screen,chunks=[asdict(c) for c in path]);proposals.append(row);all_arm.append(row)
    if screen['admitted']:selected.append((path,screen))
   # Utility is an order, never a threshold; even zero-payoff choices survive.
   selected.sort(key=lambda p:-(p[1]['payoff_utility'] or 0));cs=[c for path,_ in selected for c in path]
   expected={a['tape'] for a in all_arm if a['exact']}
   for method in PLAN['methods']:
    if time.monotonic()-start>20:raise RuntimeError('aggregate bound reached')
    r=run_search(cs,method);assert r['receipt']['status'] in ('complete_bounded_depth','skipped_no_compatible_endpoints')
    assert {o['tape'] for o in r['outputs']}==expected
    for o in r['outputs']:assert o['tape']==o['tape'][::-1]
    newruns.append(dict(arm=arm,method=method,paths=len(all_arm),early_rejected=len(all_arm)-len(selected),retained=len(selected),**r))
    k=math.ceil(.1*len(r['outputs']))
    sample.extend(dict(arm=arm,method=method,**o,review_pending=True,human_label=None) for o in rng.sample(r['outputs'],k))
  (D/'new-frame-proposals.json').write_text(json.dumps(proposals,indent=2)+'\n')
  (D/'new-frame-engine-results.json').write_text(json.dumps(newruns,indent=2)+'\n')
  (D/'sample-occurrences.json').write_text(json.dumps(sample,indent=2)+'\n')
  outputs={o['tape']:o for r in newruns for o in r['outputs']}
  summary=dict(status='complete_targeted_iteration',cached_020_paths=len(cache_records),cached_020_early_rejected=sum(not a['screen']['admitted'] for a in cache_records),cached_020_prior_states=sum(r['prior_receipt']['states_visited'] for r in replays),cached_020_filtered_states=sum(r['receipt']['states_visited'] for r in replays),cached_020_prior_failures=sum(r['prior_receipt']['failures'] for r in replays),cached_020_filtered_failures=sum(r['receipt']['failures'] for r in replays),changed_frame_proposals=len(proposals),changed_frame_early_rejected=sum(not a['screen']['admitted'] for a in proposals),changed_frame_exact_unique=len(outputs),new_to_cached_outputs=sum(t not in known for t in outputs),outputs=list(outputs.values()),sampled_occurrences=len(sample),elapsed_seconds=time.monotonic()-start,quality_claim='pending frozen-rubric model review; no human grades',scope='Small finite fixed-frame comparison; not a broad speedup or world-originality claim.')
  (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='outputs'}));print(json.dumps([o['text'] for o in outputs.values()]))
 finally:(D/'active.lock').unlink()
if __name__=='__main__':run()
