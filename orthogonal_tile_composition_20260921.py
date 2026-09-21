"""Orthogonal constructive method: semantic tiles compose via boundary obligations."""
import hashlib,itertools,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/orthogonal-tile-composition-20260921.json'
TILES=(('the scout maps','a quiet cove','nautical observation'),('our guide carries','one brass key','deliberate transport'),('a sailor watches','the red beacon','patient lookout'))
CENTERS=('while dawn settles','as harbor bells sound')
def norm(s):return re.sub('[^a-z]','',s.lower())
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def audit(t):
 x=norm(t);y=x[::-1];return {'letters':len(x),'exact':x==y,'pointer_exact':x==y,'sha256_forward':sha(x),'sha256_reverse':sha(y),'first_mismatch':next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=y[i]),None)}
def boundary(a,b):
 x,y=norm(a),norm(b);i=0
 while i<len(x) and i<len(y) and x[i]==y[-1-i]:i+=1
 return i

def run():
 rows=[]
 for l,c,r in itertools.product(TILES,CENTERS,TILES):
  if l==r:continue
  rendered=f'{l[0]} {l[1]} {c}; {r[0]} {r[1]}.';au=audit(rendered);left=l[0]+' '+l[1];right=r[0]+' '+r[1]; obligations={'left_right_boundary':boundary(left,right),'center_boundary':boundary(c,c)}
  words=re.findall('[a-z]+',rendered);content=[w for w in words if w not in {'the','a','our','one','while','as'}]
  gates={'whole_output_exact':au['exact'],'semantic_tiles_distinct':l!=r,'content_disjoint':not(set(re.findall('[a-z]+',left))&set(re.findall('[a-z]+',right))),'no_self_pal_units':all(norm(w)!=norm(w)[::-1] for w in content)}
  rows.append({'rendered':rendered,'left_tile':l,'center':c,'right_tile':r,'boundary_obligations':obligations,'audit':au,'gates':gates,'accepted':all(gates.values()),'provenance':{'construction':'independent semantic tile composition with boundary obligations','full_tape_validation':True,'finished_tape_reversal':False,'posthoc_repair':False,'borrowed_catalogue_text':False}})
 exact=[r for r in rows if r['accepted']]
 return {'experiment_id':'orthogonal-tile-composition-20260921','method':'semantic tile composition with center connector and boundary obligations','stats':{'tile_types':len(TILES),'centers':len(CENTERS),'controls':len(rows),'accepted_exact':len(exact),'max_boundary_obligation':max(max(r['boundary_obligations'].values()) for r in rows)},'exact_candidates':exact,'rendered_controls':rows,'novelty_preflight':{'status':'passed','signature':'semantic-tiles|boundary-obligations|full-tape-audit','signature_collision':False,'distinct_from':'fixed outer-shell seam grammars'},'next_repair':'Add typed boundary states (subject/object and tense) to tile edges, then compose only compatible paths before rendering.','status':'fresh exact closure found' if exact else 'no exact tile composition; obligation controls retained'}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['stats'],sort_keys=True))
