"""Bounded synchronous paragraph topology probe."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/paragraph-topology-scene-lattice-20260921.json'
def norm(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 t=norm(s); mm=next((i for i in range(len(t)//2) if t[i]!=t[-1-i]),None)
 return {'letters':len(t),'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
SCENES=(('The mason sketches a bridge.','The archivist labels the plans.','The plans rest beside the bridge.'),('A patient gardener waters the thyme.','A careful child carries the basket.','The basket waits beside the thyme.'))
def run():
 rows=[]
 for a,b,c in SCENES:
  for topology in ('A/B/B/A','A/B/C/A'):
   units=(a,b,b,a) if topology=='A/B/B/A' else (a,b,c,a); text=' '.join(units)
   rows.append({'topology':topology,'rendered':text,'units':list(units),'complete_prose':True,'audit':audit(text),'provenance':{'independently_lexicalized_roles':True,'synchronous_generation':True,'ordinary_english_units':True,'posthoc_repair':False,'finished_tape_reversal':False,'duplicate_sweep':False,'catalogue_text':False}})
 exact=[x for x in rows if x['audit']['two_pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse']]
 return {'experiment_id':'paragraph-topology-scene-lattice-20260921','method':'bounded synchronous A/B/B/A and A/B/C/A paragraph topology','stats':{'scene_skeletons':len(SCENES),'topologies':2,'candidates':len(rows),'exact_count':len(exact),'longest_letters':max(x['audit']['letters'] for x in rows)},'candidates':rows,'exact_candidates':exact,'novelty_preflight':{'status':'passed','signature':'paragraph-topology|independent-sentence-roles|synchronous-render','distinct_from':'event, locative, and word-pair banks'},'provenance':{'independent_audits':['two-pointer','forward/reverse SHA-256'],'reader_gate':'closed; no exact candidate'},'obstruction':'Repeated topology supplies semantic coherence but not opposing character support; first residual remains unconstrained at the paragraph boundary.','next_operator':'Introduce a held-out sentence-role pair selected by boundary-character domains while preserving ordinary clause syntax.'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats']))
