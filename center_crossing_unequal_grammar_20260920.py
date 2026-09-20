"""Sentence-level typed grammar with unequal arms and center-word crossing."""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parent/'runs/center-crossing-unequal-grammar-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
LEFT=('the baker carries bread','a teacher reads letters','our sailor watches stars','the gardener waters roses')
RIGHT=('the nurse answers gently','a traveler walks home','our neighbor sings softly','the painter works quietly')
def run():
 rows=[];states=buffer_pass=0
 for l in LEFT:
  for r in RIGHT:
   states+=1;lt,rt=n(l),n(r)
   # Unequal arms are intentional. The two-character buffer crosses the seam
   # inside the center words; only complete forward clauses are then rendered.
   width=2
   if len(lt)==len(rt):continue
   if lt[-width:]!=rt[:width][::-1]:continue
   buffer_pass+=1;text=l+'; '+r+'.';rows.append({'rendered':text,'left_clause':l,'right_clause':r,'audit':audit(text),'center_buffer':{'left_suffix':lt[-width:],'reverse_right_prefix':rt[:width][::-1],'width':width,'matched':True},'provenance':{'grammar':'typed subject/verb/object clause','left':'fresh forward authored clause','right':'fresh independent forward authored clause','unequal_arm_lengths':True,'center_crossing_inside_word':True,'finished_tape_reversal':False,'post_hoc_repair':False,'mirrored_units':False,'repeated_units':False,'fragment':False}})
 ex=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'center-crossing-unequal-grammar-20260920','method':'unequal typed clause grammar with live two-character center buffer','stats':{'left_clauses':len(LEFT),'right_clauses':len(RIGHT),'states':states,'unequal_center_buffer_survivors':buffer_pass,'rendered_candidates':len(rows),'fresh_exact_gt38':len(ex),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows,'exact_candidates':ex,'next_construction':'carry a variable-length residual buffer through the center word boundary while adding independently authored adjuncts','status':'fresh exact >38 candidate requires human reading' if ex else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
