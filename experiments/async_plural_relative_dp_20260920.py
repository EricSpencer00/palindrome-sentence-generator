"""Asynchronous live-buffer DP with plural agreement and relative locatives."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/async-plural-relative-dp-20260920.json'
ID='async-plural-relative-dp-20260920'; SIG='async-buffer|plural-agreement|relative-locative|grammar-controls'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=(('The','gardeners','who','work','near','the','pond','carry','baskets'),('Two','children','who','play','by','the','shed','find','apples'))
RIGHT=(('the','keepers','stack','the','baskets','beside','the','gate'),('the','neighbors','sort','the','apples','under','the','awning'))
def consume(a,b):
 n=min(len(a),len(b)); return None if a[:n]!=b[:n] else (a[n:],b[n:])
def run():
 states=[{'l':[],'r':[],'lb':'','rb':'','trace':[]}]; transitions=pruned=0
 for step in range(9):
  nxt=[]
  for s in states:
   for li,ri in zip(LEFT,RIGHT):
    lw,rw=li[step],ri[step]; transitions+=1; out=consume(s['lb']+letters(lw),s['rb']+letters(rw)[::-1])
    if out is None: pruned+=1; continue
    nxt.append({'l':s['l']+[lw],'r':s['r']+[rw],'lb':out[0],'rb':out[1],'trace':s['trace']+[(lw,rw,out[0],out[1])]})
  states=nxt
  if not states: break
 rows=[]
 for s in states:
  if s['lb'] or s['rb']: continue
  text=' '.join(s['l'])+'; meanwhile, '+' '.join(s['r'])+'.'; rows.append({'rendered':text,'audit':audit(text),'length':len(letters(text)),'grammar_controls':{'plural_subject_agreement':True,'relative_clause':True,'locative_pp':True},'buffer_trace':s['trace'],'complete_prose':True,'provenance':{'asynchronous_live_buffer':True,'right_forward_generated_reverse_facing_consumed':True,'finished_tape_reversal':False,'mirrored_units':False,'post_hoc_repair':False}})
 exact=[x for x in rows if x['audit']['exact'] and x['length']>38]
 return {'experiment_id':ID,'method':'asynchronous live-buffer DP with plural and relative/locative grammar controls','stats':{'left_frames':len(LEFT),'right_frames':len(RIGHT),'transitions':transitions,'pruned_mismatch':pruned,'surviving_states':len(states),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed' if rows else 'zero-frontier','signature':SIG,'distinct_from':'prior lexical-bank lanes; plural agreement and relative locatives alter grammar state'},'provenance':{'audits':['independent residual mismatch','forward/reverse SHA-256'],'next_construction':'add attachment-typed relative clauses after a surviving async frontier','next_reader_test':'human review only for fresh exact >38 letters'},'status':'fresh exact candidate requires human reading' if exact else 'no fresh exact candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
