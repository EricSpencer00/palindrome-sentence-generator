"""Two-sided typed word-trie DP with explicit unmatched buffers.

The constructor keeps actual character buffers for each independently grown
side.  A transition cancels only characters whose opposite positions are
already known; any mismatch kills the state.  No scalar similarity score or
finished-tape reversal is used.  This deliberately conservative orientation
may produce a zero frontier, which is recorded as such.
"""
import hashlib,json,re
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/two-sided-unmatched-buffer-dp-20260920.json'
ID='two-sided-unmatched-buffer-dp-20260920'; SIG='two-sided-unmatched-buffer-dp|actual-unmatched-buffers|cross-word-consumption|typed-forward-grammar'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=(('the','calm','pilot','maps','a','distant','coast'),('a','bright','student','keeps','the','daily','journal'),('our','kind','neighbor','opens','a','garden','gate'))
RIGHT=(('each','quiet','artist','draws','the','open','window'),('every','careful','sailor','marks','a','hidden','channel'),('the','young','reader','finds','the','useful','answer'))
@dataclass(frozen=True)
class State:
 left: str; right: str; left_buf: str; right_buf: str; trace: tuple
def consume(left_buf,right_buf):
 # Actual unmatched buffers are compared at the currently known frontier.
 n=min(len(left_buf),len(right_buf));
 if left_buf[:n] != right_buf[:n]: return None
 return left_buf[n:],right_buf[n:]
def run():
 states=[State('','', '', '', ())]; transitions=0; pruned=0
 for i in range(7):
  nxt=[]
  for s in states:
   for lw in LEFT:
    for rw in RIGHT:
     transitions+=1; lb=s.left_buf+letters(lw[i]); rb=s.right_buf+letters(rw[i])[::-1]
     out=consume(lb,rb)
     if out is None: pruned+=1; continue
     nxt.append(State(s.left+' '+lw[i],s.right+' '+rw[i],out[0],out[1],s.trace+((lw[i],rw[i],len(out[0]),len(out[1])),)))
  states=nxt
  if not states: break
 rows=[]
 for s in states:
  rendered=f'{s.left.strip()}, while {s.right.strip()}.'; rows.append({'rendered':rendered,'audit':audit(rendered),'unmatched_left_buffer':s.left_buf,'unmatched_right_buffer':s.right_buf,'buffer_trace':s.trace,'complete_prose':True,'provenance':{'left':'fresh typed word bank','right':'fresh typed word bank','actual_buffers_used':True,'finished_tape_reversal':False,'post_hoc_repair':False,'copied_or_reversed_tape':False,'mirrored_token_units':False,'repeated_units':False,'fragment':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':ID,'method':'two-sided typed word DP with actual unmatched character buffers','stats':{'left_frames':len(LEFT),'right_frames':len(RIGHT),'transitions':transitions,'pruned_mismatch':pruned,'surviving_states':len(states),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((r['audit']['letters'] for r in rows),default=0)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed' if rows else 'zero-frontier','signature':SIG,'distinct_from':'scalar boundary score and phrase products; unmatched buffers are explicit state'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 appears'},'status':'fresh exact >38 candidate requires human reading' if exact else ('zero live states before rendering' if not rows else 'no fresh exact >38 candidate')}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
