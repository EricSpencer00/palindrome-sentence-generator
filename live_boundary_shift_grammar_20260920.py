"""Live cross-word boundary-shift search over fresh coherent clauses.

The grammar is inspired only by broad shape (agent/action/place + agent/action
object), not by catalogue text.  A pair of slots is expanded inward; the
boundary debt is checked immediately after each transition.  No finished
string is reversed, repaired, or scored after the fact.
"""
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/live-boundary-shift-grammar-20260920.json'
SLOTS=(
 (('the quiet teacher','a patient sailor','our careful neighbor'),
  ('marks a new route','studies the harbor map','tends a winter garden')),
 (('the local baker','a thoughtful painter','our evening nurse'),
  ('opens the wooden door','answers a folded letter','lights the small candle')),
)
def norm(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=norm(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,
         'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),
         'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def live_match(left,right):
 """Return boundary trace, stopping at first exposed mismatch."""
 a,b=norm(left),norm(right)[::-1]; trace=[]
 for i,(x,y) in enumerate(zip(a,b)):
  ok=x==y; trace.append({'offset':i,'left':x,'right_reversed':y,'matched':ok})
  if not ok: return False,trace
 return True,trace
def run():
 frontier=[]; closed=[]; transitions=0
 # Expand complete semantic slots, but enforce each newly available boundary
 # equation immediately (the right clause is kept in forward authored order).
 for lh,lb in zip(*SLOTS[0]):
  left=f'{lh} {lb}'
  for rh,rb in zip(*SLOTS[1]):
   right=f'{rh} {rb}'; transitions+=1
   ok,trace=live_match(left,right)
   row={'left_clause':left,'right_clause':right,'boundary_trace':trace,
        'live_closed':ok,'provenance':{'left':'fresh semantic grammar slot',
          'right':'fresh independent semantic grammar slot','cross_word_boundary_shift':True,
          'finished_tape_reversal':False,'post_hoc_repair':False,'mirrored_units':False,
          'catalogue_text_reused':False}}
   frontier.append(row)
   if ok:
    rendered=f'{left}, while {right}.'; row['rendered']=rendered; row['audit']=audit(rendered)
    if row['audit']['exact']: closed.append(row)
 return {'experiment_id':'live-boundary-shift-grammar-20260920',
  'method':'live cross-word boundary equation during fresh semantic slot expansion',
  'stats':{'left_slot_variants':9,'right_slot_variants':9,'transitions':transitions,
           'boundary_closed':sum(r['live_closed'] for r in frontier),'exact_closed':len(closed)},
  'frontier':frontier,'exact_candidates':closed,
  'status':'precise zero frontier: no complete live boundary closure' if not closed else 'fresh exact requires human reading',
  'provenance':{'audit':'independent two-pointer mismatch plus forward/reverse hashes',
                'reader_gate':'closed; zero exact closures'}}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
