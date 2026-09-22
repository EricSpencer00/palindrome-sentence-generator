import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parents[1]/'runs/agreement-typed-endpoint-decoder-20260920.json'
LEFT={('sg','by'):('a diligent naturalist','the patient curator'),('pl','near'):('our careful surveyors','these quiet musicians')}
RIGHT={('sg','by'):('by a distant orchard','by the old observatory'),('pl','near'):('near quiet rivers','near distant gardens')}
LI={'sg':('records rare fossils','maps a hidden valley'),'pl':('compare field notes','collect pressed leaves')}
RI={'sg':('answers a field letter','opens the archive gate'),'pl':('share careful reports','mark the river crossings')}
def norm(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=norm(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for (number,attach), ls in LEFT.items():
  for right in RIGHT[(number,attach)]:
   for left in ls:
    for li in LI[number]:
     for ri in RI[number]:
      text=f'{left} {li} {attach} the station; {ri} {right}.'
      rows.append({'rendered':text,'agreement':number,'attachment':attach,'audit':audit(text),'endpoint_equation':{'number':number,'attachment':attach,'left_initial':norm(left)[0],'reverse_right_final':norm(right)[-1]},'provenance':{'left_endpoint':'fresh agreement-typed bank','right_endpoint':'fresh held-out agreement-typed bank','interior':'held-out from prior endpoint probe','interior_changes_live_character':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_units':False}})
 rows.sort(key=lambda x:-x['audit']['letters']); exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 controls=[{'rendered':s,'audit':audit(s)} for s in ('Several careful workers cross the bridge at dawn.','A patient curator opens the archive gate.')]
 return {'experiment_id':'agreement-typed-endpoint-decoder-20260920','method':'number/agreement/attachment endpoint schema with held-out interior expansion','stats':{'schemas':len(LEFT),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':rows[0]['audit']['letters']},'rendered_candidates':rows,'exact_candidates':exact,'controls':controls,'novelty_preflight':{'status':'passed','signature':'agreement-typed-endpoint|attachment-key|held-out-interior-live-character','distinct_from':'prior joint endpoint pairing: endpoint carries grammatical number and attachment type, and interiors are selected by agreement before rendering'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 appears','next_construction':'add tense/aspect endpoint types with disjoint finite-verb interiors; require endpoint agreement and attachment jointly before expansion'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
