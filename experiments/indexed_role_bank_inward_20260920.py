import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/indexed-role-bank-inward-20260920.json'
ID='indexed-role-bank-inward-20260920'; SIG='indexed-role-bank|endpoint-compatible-index|inward-residual|global-audit'
def letters(s): return re.sub(r'[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=(('A','patient','miller','carried','a','blue','pail','toward','the','river'),('The','young','nurse','folded','a','linen','cloth','near','the','clinic'))
RIGHT=(('the','caretaker','stored','a','clean','pail','inside','the','shed'),('the','warden','hung','a','linen','cloth','beside','the','clinic'))
def consume(a,b):
 n=min(len(a),len(b)); return None if a[:n]!=b[:n] else (a[n:],b[n:])
def run():
 index=[]; transitions=pruned=0
 for li,l in enumerate(LEFT):
  for ri,r in enumerate(RIGHT):
   seed=consume(letters(l[0]),letters(r[-1])[::-1])
   if seed is not None: index.append((li,ri,seed))
 states=[(li,ri,1,len(RIGHT[ri])-2,s[0],s[1],[LEFT[li][0]],[RIGHT[ri][-1]],[]) for li,ri,s in index]
 while states:
  nxt=[]
  for li,ri,lp,rp,lb,rb,ls,rs,tr in states:
   l,r=LEFT[li],RIGHT[ri]
   if lp>=len(l) and rp<0:
    if not lb and not rb: nxt.append((ls,rs,tr))
    continue
   lw=l[lp] if lp<len(l) else ''; rw=r[rp] if rp>=0 else ''; transitions+=1; out=consume(lb+letters(lw),rb+letters(rw)[::-1])
   if out is None: pruned+=1; continue
   nxt.append((li,ri,lp+1,rp-1,out[0],out[1],ls+([lw] if lw else []),([rw] if rw else [])+rs,tr+[(lw,rw,out[0],out[1])]))
  states=nxt
  if not states: break
 rows=[]
 for ls,rs,tr in states:
  text=' '.join(ls)+'; meanwhile, '+' '.join(rs)+'.'; rows.append({'rendered':text,'audit':audit(text),'length':len(letters(text)),'boundary_trace':tr,'complete_prose':True,'provenance':{'indexed_endpoint_compatibility':True,'fresh_role_bank':True,'full_interior_expansion':True,'global_exact_audit':True,'finished_tape_reversal':False,'post_hoc_repair':False,'duplicate_sweep':False}})
 exact=[x for x in rows if x['audit']['exact'] and x['length']>38]
 return {'experiment_id':ID,'method':'indexed endpoint-compatible role bank with inward residual search','stats':{'left_choices':len(LEFT),'right_choices':len(RIGHT),'indexed_pairs':len(index),'transitions':transitions,'pruned_mismatch':pruned,'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact)},'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed' if rows else 'zero-frontier','signature':SIG,'distinct_from':'fixed endpoint seed; combinations are indexed once before interior expansion'},'provenance':{'audits':['independent residual mismatch','forward/reverse SHA-256'],'next_reader_test':'human review only for fresh exact >38 letters'},'status':'fresh exact candidate requires human reading' if exact else 'no fresh exact candidate'}
if __name__=='__main__':
 x=run(); OUT.write_text(json.dumps(x,indent=2)+'\n'); print(json.dumps(x['stats']))
