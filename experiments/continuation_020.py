"""Prebounded continuation of 019; whole chunks, soft repetition, no API.
The earlier 36 paths are recognized and reused, not counted as discoveries.
"""
import sys,json,itertools,time,math,hashlib
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from llm_palindrome.bilateral_seams import Chunk,AllChunkCentersGrammar,OwedSideBilateralGrammar,norm
D=ROOT/'research/block-seams/continuation-020'
PREPS=('on','in','near','by','beside','inside')
MODS=('cedar','wooden','birch','narrow','sturdy','small','old','carved','clear','cargo','cold')
USES=('trade','cargo','touring','river','rescue','fishing','war','freight','travel','recreational')
HEADS=('canoe','boat','barge','raft')
PANIC_PREPS=('at','near','by','beside','inside','behind','in')
PANIC_MODS=('quiet','old','small','wooden','narrow','dark','empty','red','silent','stone')
PANIC_HEADS=('altar','church','chapel','aisle','door','window','porch','organ','font','pew','room','bench')
SOURCE='Authored finite continuation 020; lexical compounds are grammatical proposals, not independently attested semantic claims.'

def inventory(arm):
 cs=[]
 if arm.startswith('geese'):
  cs=[Chunk('L','Do geese ','START','PP','question subject; optional locative modifier',SOURCE)]
  cs += [Chunk(f'P-{p}-{m}',f"{p} {'an' if m[0] in 'aeiou' else 'a'} {m} ",'PP','NP','locative preposition, determiner, attributive modifier',SOURCE) for p,m in itertools.product(PREPS,MODS)]
  if arm=='geese_compound':
   cs += [Chunk(f'N-{u}-{h}',f'{u} {h} ','NP','VP','noun compound completing locative object',SOURCE) for u,h in itertools.product(USES,HEADS)]
   cs += [Chunk('R','see God?','VP','END','base VP under do',SOURCE)]
  else:cs += [Chunk(f'R-{h}',f'{h} see God?','NP','END','noun head closes PP, followed by question base VP',SOURCE) for h in HEADS]
 else:
  cs=[Chunk('L','We panic ','START','PP','finite clause allowing event locative adjunct',SOURCE)]
  # Determiner agreement is linked to the adjective's initial vowel.
  for article in ('a','an'):
   cs += [Chunk(f'P-{p}-{article}',f'{p} {article} ','PP',f'NP-{article}','locative preposition and determiner',SOURCE) for p in PANIC_PREPS]
  for m,h in itertools.product(PANIC_MODS,PANIC_HEADS):
   article='an' if m[0] in 'aeiou' else 'a'
   cs.append(Chunk(f'N-{m}-{h}',f'{m} {h} ',f'NP-{article}','TAIL','adjective and noun complete PP object',SOURCE))
  cs.append(Chunk('R','in a pew.','TAIL','END','second event locative; compatibility needs semantic review',SOURCE))
 assert all(len(c.text.split())>=2 for c in cs)
 return cs

def proposals(arm):
 if arm=='geese_compound':
  for p,m,u,h in itertools.product(PREPS,MODS,USES,HEADS):yield f"Do geese {p} {'an' if m[0] in 'aeiou' else 'a'} {m} {u} {h} see God?"
 elif arm=='geese_ordinary':
  for p,m,h in itertools.product(PREPS,MODS,HEADS):yield f"Do geese {p} {'an' if m[0] in 'aeiou' else 'a'} {m} {h} see God?"
 else:
  for p,m,h in itertools.product(PANIC_PREPS,PANIC_MODS,PANIC_HEADS):
   article='an' if m[0] in 'aeiou' else 'a'
   yield f'We panic {p} {article} {m} {h} in a pew.'

PLAN=dict(arms=['geese_compound','geese_ordinary','panic_nonsees'],methods=['whole_chunk_centers','owed_outside_in'],max_steps=4,max_states=10000,max_outputs=100,seconds_per_arm_method=2,aggregate_seconds=20,max_seeds=5000,seed=19030,paid_calls=0,word_fragments=False,scoring='Frozen 019 grammar/readability/coherence 0-5, holistic 0-10; model labels only; human feedback null',sampling='ceil(10% per method/arm outputs); known baseline grades reused, not relabeled',prior_attempts_reused=36)

def run():
 if (D/'summary.json').exists():raise RuntimeError('completed receipt exists; do not rerun')
 D.mkdir(parents=True,exist_ok=True);lock=D/'active.lock'
 with lock.open('x') as f:f.write('continuation_020 sole owner\n')
 try:
  (D/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n')
  prior=json.loads((ROOT/'research/block-seams/constructive-019/results/attempts.json').read_text())
  reused={norm(a['text']):a['exact'] for a in prior};old_exact={t for t,v in reused.items() if v}
  assert len(reused)==36 and len(old_exact)==1
  started=time.monotonic();records=[];counts=[];sample=[]
  for arm in PLAN['arms']:
   cs=inventory(arm);attempts=[]
   for text in proposals(arm):
    tape=norm(text);cached=tape in reused
    exact=reused[tape] if cached else tape==tape[::-1]
    attempts.append(dict(text=text,tape=tape,exact=exact,reused_prior=cached,quality_unrated=not exact))
   expected={a['tape'] for a in attempts if a['exact']};(D/f'{arm}-attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
   (D/f'{arm}-inventory.json').write_text(json.dumps([asdict(c) for c in cs],indent=2)+'\n')
   for method,cls in [('whole_chunk_centers',AllChunkCentersGrammar),('owed_outside_in',OwedSideBilateralGrammar)]:
    if time.monotonic()-started>PLAN['aggregate_seconds']:raise RuntimeError('aggregate CPU wall budget reached')
    g=cls(cs,max_seeds=PLAN['max_seeds']) if cls is AllChunkCentersGrammar else cls(cs)
    result=g.search(max_steps=4,max_states=10000,max_outputs=100,seconds=2)
    found={o['tape'] for o in result['outputs']};assert found<=expected
    if result['receipt']['status']=='complete_bounded_depth':assert found==expected
    records.append(dict(arm=arm,method=method,**result))
    k=math.ceil(.1*len(result['outputs']))
    for o in sorted(result['outputs'],key=lambda x:x['id'])[:k]:
     known=o['tape'] in old_exact
     sample.append(dict(arm=arm,method=method,**o,review=dict(reused_019_grade=known,grammar=4 if known else None,readability=3.5 if known else None,coherence=3.5 if known else None,overall_10=5.5 if known else None,human_label=None,provenance='019 known-scaffold construction' if known else '020 new proposal; awaiting model/human review')))
   counts.append(dict(arm=arm,paths=len(attempts),new_attempts=sum(not a['reused_prior'] for a in attempts),exact_paths=len(expected),new_exact_paths=len(expected-old_exact),chunks=len(cs)))
  (D/'raw-engine-results.json').write_text(json.dumps(records,indent=2)+'\n')
  (D/'sample-review.json').write_text(json.dumps(sample,indent=2)+'\n')
  summary=dict(status='complete_bounded_iteration',counts=counts,method_receipts=[dict(arm=r['arm'],method=r['method'],**r['receipt']) for r in records],new_exact_outputs=sum(c['new_exact_paths'] for c in counts),sampled_occurrences=len(sample),sample_note='Only existing 019 candidate returned; its frozen model review reused for both methods. No new or human scores.',elapsed_seconds=time.monotonic()-started)
  (D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
 finally:lock.unlink()
if __name__=='__main__':run()
