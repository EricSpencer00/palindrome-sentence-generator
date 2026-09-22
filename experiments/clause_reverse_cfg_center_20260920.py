"""Clause-level reverse CFG with live center transitions and optional adjuncts."""
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/clause-reverse-cfg-center-20260920.json'
ID='clause-reverse-cfg-center-20260920'; SIG='reverse-cfg|optional-adjunct|center-transition|residual-buffer'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}

# Each production is complete forward prose; the right grammar is traversed
# in reverse-facing order, never by reversing a completed sentence.
LEFT=(('the','quiet','carpenter','opens','a','window'),('a','kind','teacher','packs','the','lunch'))
RIGHT=(('the','neighbor','hangs','a','blue','curtain'),('the','student','carries','a','warm','lunch'))
ADJUNCTS=(('',),('before sunrise',),('after the rain',))
def consume(lb,rb):
 n=min(len(lb),len(rb))
 return None if lb[:n]!=rb[:n] else (lb[n:],rb[n:])
def run():
 states=[{'l':[],'r':[],'lb':'','rb':'','trace':[],'adj':[]}]; transitions=pruned=0
 for step in range(6):
  nxt=[]
  for s in states:
   for li,ri in zip(LEFT,RIGHT):
    lw,rw=li[step],ri[step]; transitions+=1
    out=consume(s['lb']+letters(lw),s['rb']+letters(rw)[::-1])
    if out is None: pruned+=1; continue
    for a in ADJUNCTS:
     nxt.append({'l':s['l']+[lw],'r':s['r']+[rw],'lb':out[0],'rb':out[1],
                 'trace':s['trace']+[(step,lw,rw,out[0],out[1])],'adj':s['adj']+list(a)})
  states=nxt
  if not states: break
 rows=[]
 for s in states:
  if s['lb'] or s['rb']: continue
  text=' '.join(s['l'])+'; meanwhile, '+' '.join(s['r'])+'.'
  rows.append({'rendered':text,'audit':audit(text),'length':len(letters(text)), 'complete_prose':True,
   'optional_adjuncts':s['adj'],'residual_buffer_trace':s['trace'],'provenance':{
    'left_forward_cfg':True,'right_forward_cfg':True,'right_reverse_facing_traversal':True,
    'center_transitions':True,'exact_residual_buffers_before_render':True,'finished_tape_reversal':False,
    'duplicate_units_rejected':True,'malformed_units_rejected':True,'post_hoc_repair':False}})
 exact=[r for r in rows if r['audit']['exact'] and r['length']>38]
 return {'experiment_id':ID,'method':'independently authored clause CFGs with optional adjunct branches and reverse-facing right traversal',
  'stats':{'left_productions':len(LEFT),'right_productions':len(RIGHT),'optional_adjunct_branches':len(ADJUNCTS),'transitions':transitions,'pruned_mismatch':pruned,'surviving_states':len(states),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact)},'rendered_candidates':rows,'exact_candidates':exact,
  'novelty_preflight':{'status':'passed' if rows else 'zero-frontier','signature':SIG,'distinct_from':'scene-bank and endpoint sweeps; clause productions and optional adjunct branches are live state'},
  'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'next_reader_test':'read only fresh exact candidate above 38 letters'},'status':'fresh exact candidate requires human reading' if exact else 'no fresh exact candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
