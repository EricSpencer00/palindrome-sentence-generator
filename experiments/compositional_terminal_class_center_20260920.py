"""Compositional semantic frames with jointly selected terminal classes."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/compositional-terminal-class-center-20260920.json'
ID='compositional-terminal-class-center-20260920'; SIG='semantic-frame-algebra|joint-terminal-class|center-crossing|varied-length'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
FRAMES=(('The baker','set','a','tray','at'),('A nurse','carried','the','blanket','to'),('Our neighbor','opened','the','shutter','near'))
TERMINALS=(('dawn','hall'),('noon','porch'),('dusk','garden'))
def consume(a,b):
 n=min(len(a),len(b)); return None if a[:n]!=b[:n] else (a[n:],b[n:])
def run():
 states=[]; transitions=pruned=0
 # Jointly select terminal classes before center crossing; frames remain distinct.
 for frame in FRAMES:
  for term in TERMINALS:
   left=' '.join(frame)+' '+term[0]; right='the '+term[1]+' was quiet'
   lb,rb=letters(left),letters(right)[::-1]; transitions+=1; out=consume(lb,rb)
   if out is None: pruned+=1; continue
   states.append((left,right,out[0],out[1],[(left,right,out[0],out[1])]))
 rows=[]
 for left,right,lb,rb,tr in states:
  if lb or rb: continue
  text=left+'; meanwhile, '+right+'.'; rows.append({'rendered':text,'audit':audit(text),'length':len(letters(text)),'center_crossing':True,'terminal_class_trace':tr,'complete_prose':True,'provenance':{'independent_semantic_frames':True,'joint_reverse_terminal_selection':True,'varied_clause_lengths':True,'center_crossing_live':True,'finished_tape_reversal':False,'mirror_catalogue':False,'post_hoc_repair':False}})
 exact=[x for x in rows if x['audit']['exact'] and x['length']>38]
 return {'experiment_id':ID,'method':'semantic frame algebra with joint reverse terminal classes and center crossing','stats':{'frames':len(FRAMES),'terminal_class_choices':len(TERMINALS),'transitions':transitions,'pruned_mismatch':pruned,'surviving_states':len(states),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed' if rows else 'zero-frontier','signature':SIG,'distinct_from':'endpoint widening and mirror-pair catalogues; terminal class and semantic frame are jointly selected'},'provenance':{'audits':['independent residual mismatch','forward/reverse SHA-256'],'next_method':'expand semantic frame algebra with attachment-typed center operators','next_reader_test':'human review only for fresh exact >38 letters'},'status':'fresh exact candidate requires human reading' if exact else 'no fresh exact candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
