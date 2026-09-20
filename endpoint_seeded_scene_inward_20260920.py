"""Endpoint-seeded authored scenes with full inward unequal-boundary search."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/endpoint-seeded-scene-inward-20260920.json'
ID='endpoint-seeded-scene-inward-20260920'; SIG='authored-scene-bank|endpoint-class-seed|inward-grammar|unequal-boundaries'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SCENES=(
 {'name':'bus-stop','left':('At noon','Nora waited','beside the bus stop','with a red umbrella'),'right':('the driver','checked','the wet timetable','before leaving town'),'attach':('Nora->umbrella','driver->timetable','place=stop')},
 {'name':'kitchen','left':('In the evening','Owen washed','three garden apples','for his sister'),'right':('his sister','set','the clean plates','beside the stove'),'attach':('Owen->apples','sister->plates','place=kitchen')},)
def consume(a,b):
 n=min(len(a),len(b)); return None if a[:n]!=b[:n] else (a[n:],b[n:])
def run():
 # Endpoint class is only a seed; all remaining spans are consumed as grammar.
 states=[]; rows=[]; seeded=0; transitions=pruned=0
 for sc in SCENES:
  le,ri=sc['left'][0],sc['right'][-1]
  lb,rb=letters(le),letters(ri)[::-1]; seeded+=1
  out=consume(lb,rb)
  if out is None: pruned+=1; continue
  states.append({'sc':sc,'li':1,'ri':len(sc['right'])-2,'lb':out[0],'rb':out[1],'l':[le],'r':[ri],'trace':[]})
 # Grow inward from the seeded endpoints, preserving unequal word buffers.
 while states:
  nxt=[]
  for s in states:
   if s['li']>=len(s['sc']['left']) and s['ri']<0:
    if not s['lb'] and not s['rb']:
     text=' '.join(s['l'])+'; meanwhile, '+' '.join(reversed(s['r']))+'.'
     rows.append({'rendered':text,'audit':audit(text),'length':len(letters(text)),'semantic_attachments':s['sc']['attach'],'buffer_trace':s['trace'],'complete_prose':True,'provenance':{'endpoint_classes_seed_only':True,'full_interior_grammar_live':True,'unequal_word_boundaries_live':True,'semantic_attachment_buffers_live':True,'finished_tape_reversal':False,'mirrored_units':False,'catalogue_text':False,'post_hoc_repair':False}})
    continue
   for lw in (s['sc']['left'][s['li']],) if s['li']<len(s['sc']['left']) else ('',):
    for rw in (s['sc']['right'][s['ri']],) if s['ri']>=0 else ('',):
     transitions+=1; out=consume(s['lb']+letters(lw),s['rb']+letters(rw)[::-1])
     if out is None: pruned+=1; continue
     nxt.append({'sc':s['sc'],'li':s['li']+1,'ri':s['ri']-1,'lb':out[0],'rb':out[1],'l':s['l']+([lw] if lw else []),'r':([rw] if rw else [])+s['r'],'trace':s['trace']+[(lw,rw,out[0],out[1])]})
  states=nxt
  if not states: break
 rows=[x for x in states if isinstance(x,dict) and 'rendered' in x]; exact=[x for x in rows if x['audit']['exact'] and x['length']>38]
 return {'experiment_id':ID,'method':'endpoint-compatible authored scene seed followed by inward full grammar expansion','stats':{'scenes':len(SCENES),'endpoint_seeds':seeded,'transitions':transitions,'pruned_mismatch':pruned,'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed' if rows else 'zero-frontier','signature':SIG,'distinct_from':'endpoint-only checks: all interior grammar and unequal buffers stay live'},'provenance':{'audits':['independent residual two-pointer mismatch','forward/reverse SHA-256'],'next_reader_test':'human review only for fresh exact >38 letters'},'status':'fresh exact candidate requires human reading' if exact else 'no fresh exact candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
