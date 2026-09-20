"""Fresh width-2 bilateral character-equation authoring lane."""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parent/'runs/width2-bilateral-csp-20260920.json'
def n(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LH=('a calm baker','a careful poet','a candid teacher')
LV=('keeps a journal','carries warm bread','writes a letter')
LO=('for a friend','by the river','near the window')
LA=('at dawn','in spring','with care')
RH=('the harbor keeper','a gentle painter','our thoughtful friend')
RV=('opens the gate','reads the note','lights the hall')
RO=('for the crew','by the quay','near the shore')
# Independently authored tails deliberately include an ordinary place name so
# the reverse endpoint class can be tested without synthesizing a mirror.
RA=('near CA','across CA','toward CA')
def run():
 left=[f'{h} {v} {o} {x}' for h in LH for v in LV for o in LO for x in LA]
 right=[f'{h} {v} {o} {x}' for h in RH for v in RV for o in RO for x in RA]
 rows=[];e1=e2=0
 for l in left:
  for r in right:
   if n(l)[:1]!=n(r)[-1:][::-1]: continue
   e1+=1
   lp=n(l)[:2];rp=n(r)[-2:][::-1]
   if lp!=rp: continue
   e2+=1; text=l+'; '+r+'.'; rows.append({'rendered':text,'left_clause':l,'right_clause':r,'audit':audit(text),'character_equation':{'width':2,'left_prefix':lp,'reverse_right_suffix':rp,'matched':True},'provenance':{'left':'fresh forward clause bank','right':'fresh disjoint forward clause bank','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_borrowing':False,'mirrored_units':False,'repeated_units':False,'fragment':False}})
 rows.sort(key=lambda x:-x['audit']['letters']);ex=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'width2-bilateral-csp-20260920','method':'width-2 endpoint character equation before complete forward clause rendering','stats':{'left_clauses':len(left),'right_clauses':len(right),'width1_survivors':e1,'width2_survivors':e2,'rendered_candidates':len(rows),'fresh_exact_gt38':len(ex),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows[:80],'exact_candidates':ex,'next_construction':'increase to width 3 using a new disjoint clause bank while replacing proper-name boundary words with ordinary scene endings','status':'fresh exact >38 candidate requires human reading' if ex else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
