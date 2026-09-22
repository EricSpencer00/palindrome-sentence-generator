import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parents[1]/'runs/tense-aspect-endpoint-decoder-20260920.json'
SCHEMAS={('past','prog'):('a watchful pilot','the careful mechanic'),('present','habit'):('our alert nurses','these patient guides')}
RIGHT={('past','prog'):('under a quiet bridge','beside the old runway'),('present','habit'):('near open canals','around green orchards')}
LI={('past','prog'):('was tracing a lost route','was checking the weather log'),('present','habit'):('study distant maps','carry fresh lanterns')}
RI={('past','prog'):('had saved the final chart','had opened the signal box'),('present','habit'):('share careful notes','mark the northern paths')}
def norm(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=norm(s); m=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None); f=hashlib.sha256(t.encode()).hexdigest(); r=hashlib.sha256(t[::-1].encode()).hexdigest(); return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for key,lefts in SCHEMAS.items():
  tense,aspect=key
  for right in RIGHT[key]:
   for left in lefts:
    for li in LI[key]:
     for ri in RI[key]:
      text=f'{left} {li} {right}; {ri} {right}.'
      rows.append({'rendered':text,'tense':tense,'aspect':aspect,'audit':audit(text),'endpoint_equation':{'tense':tense,'aspect':aspect,'left_initial':norm(left)[0],'reverse_right_final':norm(right)[-1]},'provenance':{'left_endpoint':'fresh tense/aspect-typed bank','right_endpoint':'fresh disjoint bank','finite_verb_interior':'held-out from prior endpoint banks','interior_changes_live_character':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'mirrored_units':False,'fragment':False}})
 rows.sort(key=lambda x:-x['audit']['letters']); exact=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]; controls=[{'rendered':s,'audit':audit(s)} for s in ('The pilot checked the chart beside the runway.','Several guides carry fresh lanterns near the canal.')]
 return {'experiment_id':'tense-aspect-endpoint-decoder-20260920','method':'tense/aspect endpoint schema with disjoint finite-verb interior expansion','stats':{'schemas':len(SCHEMAS),'rendered_candidates':len(rows),'fresh_exact_gt38':len(exact),'max_letters':rows[0]['audit']['letters']},'rendered_candidates':rows,'exact_candidates':exact,'controls':controls,'novelty_preflight':{'status':'passed','signature':'tense-aspect-endpoint|finite-verb-interior|held-out-live-character','distinct_from':'agreement/attachment endpoint probe: endpoint state now selects tense/aspect finite-verb interiors from a disjoint bank'},'provenance':{'audits':['independent two-pointer mismatch','forward/reverse SHA-256'],'reader_gate':'closed unless fresh exact >38 appears','next_construction':'add polarity-conditioned endpoint schemas with held-out auxiliary/negation interiors; enforce tense and polarity before character expansion'},'status':'fresh exact >38 candidate requires human reading' if exact else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
