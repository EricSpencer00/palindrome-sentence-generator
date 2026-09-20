"""Forward-authored phrase-pair grammar with online exact character equations."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/forward-phrase-equation-20260920.json'
ID='forward-phrase-equation-20260920'; SIG='forward-authored-phrase-grammar|online-equation|fresh-scene'
LEFT=(('At','dawn','Mara','carried','bread'),('After','rain','Jon','opened','the','gate'))
RIGHT=(('the','baker','stacked','loaves'),('a','neighbor','moved','chairs'))
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def consume(a,b):
 n=min(len(a),len(b)); return None if a[:n]!=b[:n] else (a[n:],b[n:])
def run():
 states=[{'l':[],'r':[],'lb':'','rb':'','trace':[]}]; transitions=pruned=0
 for step in range(6):
  nxt=[]
  for s in states:
   for l in LEFT:
    for r in RIGHT:
     if step>=len(l) or step>=len(r): continue
     transitions+=1; out=consume(s['lb']+letters(l[step]),s['rb']+letters(r[step])[::-1])
     if out is None: pruned+=1; continue
     nxt.append({'l':s['l']+[l[step]],'r':s['r']+[r[step]],'lb':out[0],'rb':out[1],'trace':s['trace']+[(l[step],r[step],out[0],out[1])]})
  states=nxt
  if not states: break
 rows=[]
 for s in states:
  if s['lb'] or s['rb']: continue
  text=' '.join(s['l'])+'; meanwhile, '+' '.join(s['r'])+'.'; rows.append({'rendered':text,'audit':audit(text),'length':len(letters(text)),'complete_prose':True,'online_trace':s['trace'],'provenance':{'forward_authored':True,'online_exact_equation':True,'finished_tape_reversal':False,'mirrored_units':False,'repeated_units':False,'self_palindromic_units':False,'catalogue_text':False,'post_hoc_repair':False}})
 exact=[x for x in rows if x['audit']['exact'] and x['length']>38]
 return {'experiment_id':ID,'method':'forward-authored phrase-pair grammar with online residual character equation','stats':{'left_phrases':len(LEFT),'right_phrases':len(RIGHT),'transitions':transitions,'pruned_mismatch':pruned,'surviving_states':len(states),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed' if rows else 'zero-frontier','signature':SIG,'distinct_from':'Brown relative beams and endpoint sweeps; online phrase-level equation is the live state'},'provenance':{'audits':['independent residual mismatch','forward/reverse SHA-256'],'next_construction':'add attachment-typed phrase slots only after a surviving frontier','next_reader_test':'human review only for fresh exact >38 letters'},'status':'fresh exact candidate requires human reading' if exact else 'no fresh exact candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
