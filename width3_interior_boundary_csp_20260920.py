"""Width-three endpoint plus one interior word-boundary equation."""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parent/'runs/width3-interior-boundary-csp-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LH=('Are quiet mornings','Are kind words','Are bright windows','Are old songs')
LV=('writes a note','keeps a map','carries warm bread','finds a lantern')
LO=('for a friend','near the garden','after the rain','at first light')
RH=('the opera guide','a patient actor','our evening host','the local usher')
RV=('draws a bow','takes a book','makes music','gives a scarf','shows a map','holds a bell')
RO=('at the opera','after the opera','before the opera','during the opera')

FUNCTION_WORDS={'a','an','the','our','for','near','after','at','with','toward','first'}

def content_overlap(left, right):
    words=lambda s:{w for w in re.findall(r'[a-z]+',s.casefold()) if len(w)>2 and w not in FUNCTION_WORDS}
    return sorted(words(left)&words(right))
def run():
 left=[(h,v,o) for h in LH for v in LV for o in LO];right=[(h,v,o) for h in RH for v in RV for o in RO]
 rows=[];w1=w3=ib=0
 for h,v,o in left:
  for rh,rv,ro in right:
   lt=n(' '.join((h,v,o)));rt=n(' '.join((rh,rv,ro)))
   if lt[:1]!=rt[-1:][::-1]:continue
   w1+=1
   if lt[:3]!=rt[-3:][::-1]:continue
   w3+=1
   # Interior word-boundary equation: first char of left's second slot
   # equals the reverse-facing last char of right's second slot.
   if n(v)[0]!=n(rv)[-1]:continue
   if n(o)[0]!=n(ro)[0]:continue
   ib+=1;text=f'{h} {v} {o}; {rh} {rv} {ro}.'
   overlap=content_overlap(text.split('; ')[0],text.split('; ')[1])
   rows.append({'rendered':text,'left_clause':text.split('; ')[0],'right_clause':text.split('; ')[1],'audit':audit(text),'equations':{'endpoint_width':3,'left_prefix':lt[:3],'reverse_right_suffix':rt[-3:][::-1],'interior_boundary':{'left_slot_prefix':n(v)[0],'reverse_right_slot_suffix':n(rv)[-1],'matched':True}},'provenance':{'left':'fresh forward clause bank','right':'fresh disjoint forward clause bank','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_borrowing':False,'mirrored_units':False,'repeated_units':bool(overlap),'repeated_content_words':overlap,'fragment':False}})
 rows.sort(key=lambda x:-x['audit']['letters']);ex=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'width3-interior-boundary-csp-20260920','method':'width-three endpoint state plus live interior word-boundary character equation','stats':{'left_clauses':len(left),'right_clauses':len(right),'width1_survivors':w1,'width3_survivors':w3,'interior_boundary_survivors':ib,'rendered_candidates':len(rows),'repeated_content_rows':sum(bool(x['provenance']['repeated_content_words']) for x in rows),'fresh_exact_gt38':len(ex),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows[:80],'exact_candidates':ex,'next_construction':'carry the matched interior boundary into a second interior slot with a new disjoint bank','status':'fresh exact >38 candidate requires human reading' if ex else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
