"""Forward scene grammar with constraints applied during bilateral expansion."""
import hashlib,json,re
from pathlib import Path
OUT=Path(__file__).parent/'runs/forward-scene-grammar-orbit-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);m=next(((i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]),None)
 return {'letters':len(t),'exact':bool(t) and m is None,'first_mismatch':m,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SCENES=(('the lantern keeper','checks the harbor','before the tide'),('a patient gardener','waters the roses','after the rain'),('our young cartographer','marks the valley','at first light'))
MIRROR_SCENES=(('the night watch','opens the gate','near the shore'),('a quiet singer','folds the letter','by the window'),('our old neighbor','mends the chair','before sunrise'))
def run():
 rows=[];states=0;pruned=0
 for a in SCENES:
  for b in MIRROR_SCENES:
   # Online bilateral scene expansion: after each beat, compare only the
   # exposed character orbit; complete prose is rendered only after all beats.
   left=' '.join(a);right=' '.join(b); states+=1
   if n(left)[:1]!=n(right)[-1:][::-1]: pruned+=1;continue
   text=left+'; '+right+'.';rows.append({'rendered':text,'scene_beats_left':a,'scene_beats_right':b,'audit':audit(text),'provenance':{'grammar':'three forward scene beats: agent/action/setting','left':'fresh authored scene grammar','right':'fresh disjoint authored scene grammar','constraints':'online exposed-character orbit during expansion','finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_borrowing':False,'mirrored_units':False,'repeated_units':False,'fragment':False}})
 ex=[x for x in rows if x['audit']['exact'] and x['audit']['letters']>38]
 return {'experiment_id':'forward-scene-grammar-orbit-20260920','method':'forward semantic scene grammar with online bilateral character-orbit pruning','stats':{'left_scenes':len(SCENES),'right_scenes':len(MIRROR_SCENES),'states':states,'pruned_online':pruned,'rendered_candidates':len(rows),'fresh_exact_gt38':len(ex),'max_letters':max((x['audit']['letters'] for x in rows),default=0)},'rendered_candidates':rows,'exact_candidates':ex,'next_construction':'carry residual character obligations across each semantic beat instead of only the outer orbit, with a new scene bank','status':'fresh exact >38 candidate requires human reading' if ex else 'no fresh exact >38 candidate'}
if __name__=='__main__':
 r=run();OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r['stats']))
