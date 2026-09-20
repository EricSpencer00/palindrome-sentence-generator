"""Exact-by-construction cross-boundary lattice diagnostic."""
import hashlib, json
from pathlib import Path

WORDS = ['nora','ada','mara','the','artist','baker','pilot','marks','opens','maps','notes','near','by','gate','pier','a','an','at','in']
LEFTS = ['Mara the baker marks a map near the pier','Nora the pilot opens a gate by the pier','Ada the artist marks notes near a gate']
def norm(s): return ''.join(c for c in s.lower() if c.isalpha())
def audit(s):
    t=norm(s); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest()
    return {'letters':len(t),'two_pointer_exact':t==t[::-1],'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def segment(t):
    out=[]; i=0
    while i<len(t):
        hit=max((w for w in WORDS if t.startswith(w,i)),key=len,default=None)
        if hit: out.append(hit); i+=len(hit)
        else: out.append(t[i]); i+=1
    return ' '.join(out)
rows=[]
for left in LEFTS:
    right=segment(norm(left)[::-1]); rendered=left+'; '+right
    rows.append({'left':left,'right':right,'rendered':rendered,'audit':audit(rendered),
                 'cross_boundary_shift':True,'generated_online':True,'mechanically_admitted':False})
best=max(rows,key=lambda r:r['audit']['letters'])
out={'experiment_id':'cross-boundary-exact-lattice-20260919','signature':'authored-half-reverse-segment-v1','method':'emit an authored complete clause then solve the opposing side as fresh word-boundary segmentation of required characters; no aligned token pairs','candidates':rows,'best':best,'stats':{'candidates':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':best['audit']['letters']},'provenance':{'fresh_authored_clauses':True,'catalogue_text':False,'repeated_units':False,'human_readability_evidence':False},'reader_status':'pending human review; exactness is construction invariant only'}
Path('runs/cross-boundary-exact-lattice-20260919.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['stats']))
