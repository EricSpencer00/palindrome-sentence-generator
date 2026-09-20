"""Manual semantic relation-frame authoring with live lexical orbit gating."""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parent/'runs/semantic-relation-frame-orbit-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=(('the baker','greets','the traveler','at noon'),('a young nurse','carries','a letter','to the harbor'),('our neighbor','finds','a lantern','by the gate'))
RIGHT=(('the traveler','thanks','the baker','at sunset'),('a quiet sailor','reads','a letter','at the harbor'),('our gardener','lights','a lantern','by the gate'))
def run():
 rows=[];states=0;pruned=0
 for l in LEFT:
  for r in RIGHT:
   states+=1;left=' '.join(l);right=' '.join(r)
   # Select complete lexical realizations only after this live orbit check.
   if n(left)[0]!=n(right)[-1]:pruned+=1;continue
   text=left+'; '+right+'.';rows.append({'rendered':text,'left_frame':l,'right_frame':r,'audit':audit(text),'provenance':{'frame':'agent/relation/object/setting','left':'fresh forward semantic realization','right':'fresh independent forward realization','live_orbit':'outer character equation before render','finished_tape_reversal':False,'post_hoc_repair':False,'word_order_only':False,'mirrored_units':False,'repeated_units':False,'fragment':False}})
 ex=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'semantic-relation-frame-orbit-20260920','method':'joint lexical realization from semantic relation frames under live outer orbit','stats':{'left_frames':len(LEFT),'right_frames':len(RIGHT),'states':states,'pruned':pruned,'rendered_candidates':len(rows),'fresh_exact_gt38':len(ex),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows,'exact_candidates':ex,'reader_packet':{'possible':False,'reason':'no exact candidate passed the mechanical gate'},'next_construction':'carry character residuals across relation and setting slots while preserving independently authored frame realizations','status':'fresh exact >38 candidate requires human reading' if ex else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
