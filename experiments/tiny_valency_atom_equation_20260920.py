"""Frozen tiny semantic atom bank; live two-sided equation diagnostic."""
import hashlib,json,re,itertools
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/tiny-valency-atom-equation-20260920.json'
def n(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=n(s); mm=next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=x[-1-i]),None)
 return {'letters':len(x),'pointer_exact':mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest()}
LEFT=[('Mira carries the lantern','carry'),('The nurse checks the pulse','check'),('A scout watches the ridge','watch'),('The baker warms the oven','warm')]
RIGHT=[('while the harbor keeps its light','carry'),('so the patient rests nearby','check'),('as the valley hears the bell','watch'),('and the bread rises slowly','warm')]
def run():
 rows=[]
 for l,r in itertools.product(LEFT,RIGHT):
  if l[1]!=r[1]: continue
  t=l[0]+'; '+r[0]+'.'; rows.append({'rendered':t,'valency':l[1],'audit':audit(t),'provenance':{'frozen_tiny_bank':True,'human_authored_atoms':True,'selected_before_rendering':True,'hidden_palindromic_span':False,'repeated_units':False,'posthoc_repair':False,'finished_tape_reversal':False}})
 exact=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse']]
 best=min(rows,key=lambda x: x['audit']['first_mismatch'][0] if x['audit']['first_mismatch'] else -1) if rows else None
 return {'experiment_id':'tiny-valency-atom-equation-20260920','method':'frozen human-authored phrase atoms with shared semantic valency and live two-sided character audit','stats':{'left_atoms':len(LEFT),'right_atoms':len(RIGHT),'valency_joins':len(rows),'exact':len(exact),'exact_gt38':sum(x['audit']['letters']>38 for x in exact)},'exact_candidates':exact,'strongest_near_miss':best,'obstruction':best['audit']['first_mismatch'] if best else None,'status':'no exact closure; tiny-bank seam obstruction retained' if not exact else 'fresh exact requires human reading'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
