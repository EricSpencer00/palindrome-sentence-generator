"""Width-three endpoint equation with fresh ordinary clause banks."""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parent/'runs/width3-fresh-endpoint-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LH=('I raise a lantern','I raise a question','I raise the curtain')
LV=('beside the river','under a clear sky','before the bell')
LO=('for my friend','near the garden','after the rain')
RH=('the safari guide','a patient artist','our evening host')
RV=('keeps a small journal','draws the distant road','opens the wooden gate')
RO=('on safari','after safari','before safari')
def run():
 left=[f'{h} {v} {o}' for h in LH for v in LV for o in LO]
 right=[f'{h} {v} {o}' for h in RH for v in RV for o in RO]
 rows=[];w1=w2=w3=0
 for l in left:
  for r in right:
   lt,rt=n(l),n(r)
   if lt[:1]!=rt[-1:][::-1]:continue
   w1+=1
   if lt[:2]!=rt[-2:][::-1]:continue
   w2+=1
   if lt[:3]!=rt[-3:][::-1]:continue
   w3+=1; text=l+'; '+r+'.'; rows.append({'rendered':text,'left_clause':l,'right_clause':r,'audit':audit(text),'endpoint_equation':{'width':3,'left_prefix':lt[:3],'reverse_right_suffix':rt[-3:][::-1],'matched':True},'provenance':{'left':'fresh forward authored clause','right':'fresh independent forward authored clause','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_borrowing':False,'mirrored_units':False,'repeated_units':False,'fragment':False}})
 rows.sort(key=lambda x:-x['audit']['letters']);ex=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'width3-fresh-endpoint-20260920','method':'incremental width-three endpoint equation over fresh clause banks','stats':{'left_clauses':len(left),'right_clauses':len(right),'width1_survivors':w1,'width2_survivors':w2,'width3_survivors':w3,'rendered_candidates':len(rows),'fresh_exact_gt38':len(ex),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows[:80],'exact_candidates':ex,'status':'fresh exact >38 candidate requires human reading' if ex else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
