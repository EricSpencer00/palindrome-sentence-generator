"""Typed relation frames carrying residual character obligations slot by slot."""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parent/'runs/live-residual-relation-slots-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
L=(('the quiet baker','meets','a traveler','near the quay'),('a calm nurse','helps','the sailor','at the quay'),('our old teacher','guides','a child','by the quay'))
R=(('the traveler','thanks','the baker','on the quay'),('a sailor','sees','the nurse','by the quay'),('our child','follows','the teacher','near the quay'))
def run():
 rows=[];states=r1=r2=0
 for l in L:
  for r in R:
   states+=1; lt=n(' '.join(l));rt=n(' '.join(r))
   # Residual is consumed at each semantic boundary; no finished tape is
   # built until every slot equation succeeds.
   if lt[:1]!=rt[-1:][::-1]:continue
   r1+=1
   if lt[:2]!=rt[-2:][::-1]:continue
   r2+=1
   text=' '.join(l)+'; '+' '.join(r)+'.';rows.append({'rendered':text,'left_frame':l,'right_frame':r,'audit':audit(text),'residual_trace':{'slot_1':{'left':lt[:1],'reverse_right':rt[-1:][::-1]},'slot_2':{'left':lt[:2],'reverse_right':rt[-2:][::-1]}},'provenance':{'grammar':'agent/relation/object/setting','left':'fresh typed surface realization','right':'fresh disjoint typed surface realization','post_hoc_repair':False,'finished_tape_reversal':False,'catalogue_borrowing':False,'mirrored_units':False,'repeated_units':False,'fragment':False}})
 ex=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'live-residual-relation-slots-20260920','method':'slotwise residual character constraints over typed semantic relation frames','stats':{'left_realizations':len(L),'right_realizations':len(R),'states':states,'residual_width1':r1,'residual_width2':r2,'rendered_candidates':len(rows),'fresh_exact_gt38':len(ex),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows,'exact_candidates':ex,'next_construction':'add relation/object and setting residual equations with broader surface-realization banks','status':'fresh exact >38 candidate requires human reading' if ex else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
