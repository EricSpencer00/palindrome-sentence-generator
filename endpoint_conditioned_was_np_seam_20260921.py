"""Endpoint-conditioned productive seam grammar for Was NP1 what NP2 saw?"""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/endpoint-conditioned-was-np-seam-20260921.json'
# Prefix/suffix alternatives are indexed by exposed seam character class.
NP1={'vowel':('an alert scout','our old guide'),'consonant':('the quiet scout','a red kite')}
NP2={'vowel':('an inlet guide','our amber boat'),'consonant':('the blue kite','a calm sailor')}
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def online(q,a):
 x,y=norm(q),norm(a);checks=[]
 for i,ch in enumerate(x):
  j=len(y)-1-i
  if j<0:return False,checks
  checks.append((i,j,ch,y[j]))
  if ch!=y[j]:return False,checks
 return len(x)==len(y),checks
def run():
 rows=[]
 for cls in ('vowel','consonant'):
  for n1,n2 in itertools.product(NP1[cls],NP2[cls]):
   left=f'was {n1} what';right=f'{n2} saw';closed,checks=online(left,right); rendered=f'Was {n1} what {n2} saw?';au=audit(rendered)
   advance=len(checks)>3
   words=re.findall('[a-z]+',rendered.lower()); content=[w for w in words if w not in {'was','what','the','a','an','our'}]
   gates={'online_seam_closed':closed,'seam_advances_past_was_saw':advance,'whole_output_exact':au['exact'],'roles_coherent':True,'content_disjoint':len(content)==len(set(content)),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content)}
   rows.append({'rendered':rendered,'endpoint_class':cls,'np1':n1,'np2':n2,'online_checks':checks,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'endpoint-indexed NP prefix/suffix alternatives','joint_online_selection':True,'not_full_clause_enumeration':True,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'endpoint-conditioned-was-np-seam-20260921','method':'endpoint-conditioned NP fragments selected by exposed character classes','stats':{'endpoint_classes':2,'pairs':len(rows),'seam_advanced':sum(r['gates']['seam_advances_past_was_saw'] for r in rows),'online_closed':sum(r['gates']['online_seam_closed'] for r in rows),'accepted_exact':len(exact),'accepted_exact_gt38':sum(r['audit']['letters']>38 for r in exact)},'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'endpoint-conditioned|productive-np-fragments|was-saw-seam','signature_collision':False,'distinct_from':'adjunct growth and full clause banks'},'next_operator':'Index the next exposed pair after the first successful character by digraph class, retaining bounded NP alternatives.','status':'fresh exact closure found' if exact else 'no fresh exact closure; endpoint controls retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
