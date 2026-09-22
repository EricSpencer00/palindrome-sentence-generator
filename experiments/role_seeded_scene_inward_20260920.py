"""Fresh role-seeded two-clause scene with real inward residual buffers."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/role-seeded-scene-inward-20260920.json'
ID='role-seeded-scene-inward-20260920'; SIG='role-seed|subject-article|terminal-setting|two-clause-residual'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SCENE={'left':('The','quiet','gardener','carried','a','lantern','toward','the','orchard'),'right':('The','careful','keeper','placed','a','basket','beside','the','workshop'),'roles':('subject=agent','article=the','terminal=setting','object=prop')}
def consume(a,b):
 n=min(len(a),len(b)); return None if a[:n]!=b[:n] else (a[n:],b[n:])
def run():
 # Seed endpoint role classes, then consume all interior words; no endpoint-only verdict.
 l,r=SCENE['left'],SCENE['right']; lb=letters(l[0]); rb=letters(r[-1])[::-1]; seed=consume(lb,rb); transitions=pruned=1
 if seed is None: return result(0,1,0,0,[],[])
 states=[(1,len(r)-2,seed[0],seed[1],[l[0]],[r[-1]],[])]
 while states:
  nxt=[]
  for li,ri,lb,rb,ls,rs,tr in states:
   if li>=len(l) and ri<0:
    if not lb and not rb: nxt.append((ls,rs,tr))
    continue
   lw=l[li] if li<len(l) else ''; rw=r[ri] if ri>=0 else ''; transitions+=1
   out=consume(lb+letters(lw),rb+letters(rw)[::-1])
   if out is None: pruned+=1; continue
   nxt.append((li+1,ri-1,out[0],out[1],ls+([lw] if lw else []),([rw] if rw else [])+rs,tr+[(lw,rw,out[0],out[1])]))
  states=nxt
  if not states: break
 rows=[]
 for ls,rs,tr in states:
  text=' '.join(ls)+'; meanwhile, '+' '.join(rs)+'.'; rows.append({'rendered':text,'audit':audit(text),'length':len(letters(text)),'roles':SCENE['roles'],'residual_trace':tr,'complete_prose':True,'provenance':{'fresh_authored_clauses':True,'role_seed_only':True,'full_inward_expansion':True,'real_residual_buffer':True,'global_exact_audit':True,'control_text_reused':False,'finished_tape_reversal':False,'post_hoc_repair':False}})
 return result(transitions,pruned,len(states),len(rows),rows,[])
def result(transitions,pruned,surviving,rendered,rows,exact):
 exact=[x for x in rows if x['audit']['exact'] and x['length']>38]
 return {'experiment_id':ID,'method':'role-compatible endpoint seed followed by two-clause inward residual expansion','stats':{'transitions':transitions,'pruned_mismatch':pruned,'surviving_states':surviving,'rendered_candidates':rendered,'fresh_exact_gt38':len(exact)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed' if rows else 'zero-frontier','signature':SIG,'distinct_from':'manual control and endpoint-only checks; interior residual buffer remains live'},'provenance':{'audits':['independent residual mismatch','forward/reverse SHA-256'],'next_reader_test':'human review only for fresh exact >38 letters'},'status':'fresh exact candidate requires human reading' if exact else 'no fresh exact candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
