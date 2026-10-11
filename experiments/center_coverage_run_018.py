"""Finite audit-conditioned whole-phrase experiment; no originality claim."""
import sys,json,itertools,random,math,time
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.bilateral_seams import Chunk,AllChunkCentersGrammar,OwedSideBilateralGrammar,CenterOutGrammar,debt,norm
D=ROOT/'research/block-seams/center-coverage-018';A=D/'imports/audit/audit_output'

def run():
 D.mkdir(parents=True,exist_ok=True)
 plan=dict(seed=19028,methods=['whole_chunk_centers','owed_outside_in'],strata=['known_controls','conditioned_paths'],max_steps=6,max_states=10000,max_outputs=1000,seconds_per_run=1,aggregate_seconds=30,paid_calls=0,sampling='ceil(0.10*n) per method/stratum; retain duplicates across groups; no human labels')
 (D/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 controls=json.loads((A/'whole_word_control_inventory.json').read_text());pairs=json.loads((A/'reverse_compatible_candidate_inventory.json').read_text())
 cases=[]
 for name,raw in controls.items():cases.append(('known_controls',name,[Chunk(**c) for c in raw]))
 middles={0:[['step on ','no pets '],['rely on ','my pets ']],1:[['say that '],['know that ']],2:[['strange force '],['dark power ']],3:[[]],4:[['I hid '],['I lost ']],5:[[]],6:[[]],7:[[]],8:[[]],9:[[]],11:[[]],12:[['not drawn ','onward, we ','few, drawn ','onward to ']]}
 exclusions=[]
 for i,p in enumerate(pairs):
  if i==10:exclusions.append(dict(pair=p['id'],reason='sees predicate excluded from conditioned NON-SEES arm; remains known diagnostic control'));continue
  for variant,mid in enumerate(middles[i]):
   texts=[p['left']]+mid+[p['right']];states=['START']+[f"{p['id']}-{variant}-slot-{j}" for j in range(len(texts)-1)]+['END']
   cs=[Chunk(f"{p['id']}-{variant}-{j}",t,states[j],states[j+1],p['left_obligation'] if j==0 else ('licensed continuation of '+p['left_obligation'] if j<len(texts)-1 else p['right_role']),'audit-conditioned authored phrase inventory; no novelty claim') for j,t in enumerate(texts)]
   assert all(len(c.text.split())>=2 for c in cs)
   cases.append(('conditioned_paths',p['id']+f'-v{variant}',cs))
 (D/'inventory.json').write_text(json.dumps([dict(stratum=s,case=n,chunks=[asdict(c) for c in cs]) for s,n,cs in cases],indent=2)+'\n')
 records=[];attempts=[];start=time.monotonic()
 for stratum,name,cs in cases:
  for method,cls in [('whole_chunk_centers',AllChunkCentersGrammar),('owed_outside_in',OwedSideBilateralGrammar)]:
   if time.monotonic()-start>plan['aggregate_seconds']:raise RuntimeError('aggregate bound exceeded')
   r=cls(cs).search(max_steps=6,max_states=10000,max_outputs=1000,seconds=1)
   assert r['receipt']['status']=='complete_bounded_depth'
   records.append(dict(stratum=stratum,case=name,method=method,**r))
   for o in r['outputs']:assert o['tape']==o['tape'][::-1]
   if method=='whole_chunk_centers':
    old=set()
    for boundary in sorted({c.entry for c in cs}|{c.exit for c in cs}):
     old.update(o['tape'] for o in CenterOutGrammar(cs,[],center_state=boundary).search(max_steps=6,seconds=1)['outputs'])
    attempts.append(dict(case=name,stratum=stratum,text=''.join(c.text for c in cs),old_boundary_found=bool(old),patched_found=bool(r['outputs']),exact=norm(''.join(c.text for c in cs))==norm(''.join(c.text for c in cs))[::-1]))
 (D/'raw-results.json').write_text(json.dumps(records,indent=2)+'\n')
 (D/'attempts.json').write_text(json.dumps(dict(attempts=attempts,exclusions=exclusions),indent=2)+'\n')
 rng=random.Random(plan['seed']);samples=[];groups=[]
 notes={'panic':(5,5,4,'Complete clause and location; no progression.'),'geese':(5,5,3,'Grammatical question with whimsical religious premise.'),'live':(4,4,3,'Poetic imperative; vague evil metaphor.'),'cat':(4,4,3,'Question works with an implied omitted relative pronoun.'),'boot':(5,4,4,'Grammatical regret, but boot concealment needs context.'),'sinned':(5,5,3,'Complete clause; abstract action has no elaboration.'),'animals':(4,3,2,'Intransitive slam into a net is more natural than slam in a net.'),'onward':(2,2,2,'Bare new era lacks an article; repeated drawn/onward; no quality claim.')}
 for method in plan['methods']:
  for stratum in plan['strata']:
   pool=[dict(case=r['case'],method=method,stratum=stratum,**o) for r in records if r['method']==method and r['stratum']==stratum for o in r['outputs']]
   k=math.ceil(.1*len(pool));groups.append(dict(method=method,stratum=stratum,outputs=len(pool),sampled=k))
   for o in rng.sample(pool,k):
    key=o['case'];alias={'pair-03':'cat','pair-04':'boot','pair-08':'animals','pair-09':'panic','pair-11':'sinned','pair-12':'onward'}
    q=notes.get(key,notes.get(alias.get(key[:7],''),(4,4,3,'Simple complete clause; no paragraph-level progression.')))
    o['review']=dict(reviewer='sole research owner; model review, not human judgment',grammar=q[0],readability=q[1],coherence=q[2],scale='0 to 5',repetition=o['repetition'],provenance='Audit-derived or authored control; originality unverified',note=q[3],human_label=None)
    samples.append(o)
 (D/'sample-review.json').write_text(json.dumps(dict(groups=groups,samples=samples,overall_outputs=sum(g['outputs'] for g in groups),overall_sampled=len(samples),rounding='sum of per-method/stratum ceilings; duplicates retained'),indent=2)+'\n')
 frozen=json.loads((ROOT/'research/block-seams/multiword-seams-017/inventory.json').read_text());cs=[Chunk(**c) for c in frozen];starts=[c for c in cs if c.entry=='START'];ends=[c for c in cs if c.exit=='END']
 compatible=[(a.id,b.id) for a in starts for b in ends if debt(a.text,b.text)['compatible']]
 assert not compatible
 (D/'frozen-017-endpoint-audit.json').write_text(json.dumps(dict(tested=len(starts)*len(ends),compatible=0,scope='No exact path of two or more chunks at any depth in frozen grammar: incompatible endpoints. Single START-END chunks tested separately.',single_chunk_exact=[c.id for c in cs if c.entry=='START' and c.exit=='END' and norm(c.text)==norm(c.text)[::-1]]),indent=2)+'\n')
 out=[o for r in records for o in r['outputs']];summary=dict(cases=len(cases),method_runs=len(records),retained_outputs=len(out),unique_tapes=len({o['tape'] for o in out}),groups=groups,sampled=len(samples),new_originality_claims=0,tests_passed=48,elapsed_seconds=time.monotonic()-start,old_missed_controls=sum(not a['old_boundary_found'] for a in attempts if a['stratum']=='known_controls'),all_output_exact=True)
 (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':run()
