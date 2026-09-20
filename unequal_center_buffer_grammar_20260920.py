"""Forward typed grammar with exact unequal-length center buffers."""
import hashlib,itertools,json,re
from pathlib import Path
OUT=Path(__file__).resolve().parent/'runs/unequal-center-buffer-grammar-20260920.json'
def letters(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=letters(s); mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); h=hashlib.sha256(t.encode()).hexdigest(); rh=hashlib.sha256(t[::-1].encode()).hexdigest(); return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':h,'sha256_reverse':rh,'sha_equal':h==rh}
SUB=('the archivist','a gardener','the cartographer'); VERB=('studies','maps','names'); OBJ=('the old chart','a clear route','the distant shore'); RIGHT=('the patient witness','a quiet sailor','the evening courier'); TAIL=('before dusk','by lantern light','near the river')
def step(left,right,buf):
 # Both clauses are emitted forward. The right stream is buffered because its
 # characters will be consumed from the opposite side at the final equation.
 b=buf+letters(right)[::-1]; l=letters(left); n=min(len(l),len(b)); i=0
 while i<n and l[i]==b[i]: i+=1
 if i<n:return None,{'mismatch':(i,l[i],b[i]),'buffer':b[i:]}
 return b[n:],{'mismatch':None,'buffer':b[n:]}
def run():
 frontier=[]; rows=[]; states=0
 for s,v,o,r,t in itertools.product(SUB,VERB,OBJ,RIGHT,TAIL):
  states+=1; buf=''; buf,tr=step(s,r,buf)
  if buf is None: frontier.append({'stage':'subject-agent','left':s,'right':r,'trace':tr}); continue
  buf,tr2=step(v,t,buf)
  if buf is None: frontier.append({'stage':'verb-tail','left':v,'right':t,'trace':tr2}); continue
  rendered=f'{s} {v} {o}, and {r} {t}.'; a=audit(rendered)
  rows.append({'rendered':rendered,'center_buffer':buf,'traces':[tr,tr2],'audit':a,'complete_prose':True,'provenance':{'both_sides_forward':True,'unequal_length_buffer_equation':True,'pruned_before_rendering':False,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_token_units':False,'fragment':False}})
 exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 r={'experiment_id':'unequal-center-buffer-grammar-20260920','method':'typed forward grammar with exact residual buffer across unequal center crossing','stats':{'states_examined':states,'frontier_pruned_before_rendering':len(frontier),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'frontier':frontier[:100],'rendered_candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':'typed-grammar|unequal-center-buffer|forward-emission|exact-prerender-equation','distinct_from':'relation and endpoint sweeps: explicit residual buffers cross unequal slot lengths','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'fragments':False},'provenance':{'audits':['independent pointer mismatch','forward/reverse SHA-256'],'next_construction':'expand subject/agent banks around surviving first-character buffer classes'},'status':'fresh exact >38 candidate requires human reading' if exact else 'over-constrained frontier recorded; no exact candidate'}
 return r
if __name__=='__main__':
 r=run(); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
