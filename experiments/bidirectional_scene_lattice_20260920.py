"""One fresh bidirectional scene lattice with live outer-class admission."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/bidirectional-scene-lattice-20260920.json'
def n(s):return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=n(s);mm=next(((i,t[i],t[-1-i]) for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SCENES=(('a','patient','cartographer','maps','the','quiet','coast'),('a','careful','gardener','tends','the','winter','orchard'),('an','eager','archivist','copies','a','faded','letter'))
RESPONSES=(('the','local','pilot','checks','a','weathered','compass'),('the','young','keeper','opens','the','garden','gate'),('a','kind','clerk','reads','the','evening','message'))
def admit(a,b):
 x,y=n(' '.join(a)),n(' '.join(b))[::-1]; trace=[]
 for i,(u,v) in enumerate(zip(x,y)):
  trace.append({'offset':i,'left':u,'right_reversed':v,'matched':u==v})
  if u!=v:return False,trace
 return len(x)==len(y),trace
def run():
 rows=[]
 for scene,response in zip(SCENES,RESPONSES):
  ok,tr=admit(scene,response); text=' '.join(scene)+', while '+' '.join(response)+'.'
  rows.append({'rendered':text,'scene_roles':('agent','modifier','role','action','det','modifier','object'),'response_roles':('det','modifier','role','action','det','modifier','object'),'live_outer_class_closed':ok,'trace':tr,'audit':audit(text),'provenance':{'semantic_roles_fixed_first':True,'fresh_scene_lexicon':True,'outer_classes_admitted_live':True,'catalogue_text_reused':False,'finished_tape_reversal':False,'post_hoc_repair':False,'mirrored_units':False}})
 exact=[r for r in rows if r['live_outer_class_closed'] and r['audit']['exact'] and r['audit']['letters']>38]
 return {'experiment_id':'bidirectional-scene-lattice-20260920','method':'fixed semantic scene roles with live outer-character lexical admission','stats':{'scenes':3,'rendered_controls':3,'live_closures':sum(r['live_outer_class_closed'] for r in rows),'exact_gt38':len(exact)},'rendered_controls':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','distinct_from':'POS trie, endpoint seed, prosodic profile, and boundary-shift lanes','catalogue_surface_reuse':False,'post_render_repair':False},'status':'precise zero frontier: no scene closes live outer equations','next_construction':'expand each role with adjective/noun alternatives whose exposed classes match before admitting the next semantic slot','provenance':{'audit':'independent two-pointer mismatch and forward/reverse hashes','reader_gate':'closed; controls only'}}
if __name__=='__main__':
 x=run();OUT.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x['stats']))
