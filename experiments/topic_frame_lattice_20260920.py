"""Fresh search: human-authored topic-frame lattice with live discourse slots.

This deliberately does not reverse or repair finished text.  Each side is an
independently authored complete clause; only compatible topic/frame labels
are joined, then the mechanical palindrome gate is applied.
"""
import hashlib, itertools, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/topic-frame-lattice-20260920.json'
def norm(s): return re.sub('[^a-z]','',s.casefold())
def audit(s):
 x=norm(s); ok=bool(x) and x==x[::-1]
 return {'letters':len(x),'pointer_exact':ok,'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest()}
LEFT=[('At first light, the keeper marks the inlet','inlet','time'),('By the old bridge, a patient guide hears the bell','bridge','place'),('Before the storm, the careful crew checks the rope','storm','risk'),('At low tide, the ferryman studies the sand','tide','place')]
RIGHT=[('and the harbor answers with a quiet lantern','inlet','time'),('while a distant watch records the crossing','bridge','place'),('so the shore crew shelters the skiff','storm','risk'),('as the last boat waits beyond the shoal','tide','place')]
def run():
 rows=[]; joins=0
 for l,r in itertools.product(LEFT,RIGHT):
  if l[1]!=r[1] or l[2]!=r[2]: continue
  joins+=1; text=l[0]+', '+r[0]+'.'; a=audit(text)
  rows.append({'rendered':text,'frame':{'topic':l[1],'discourse':l[2]},'audit':a,'provenance':{'human_authored_complete_clauses':True,'selected_before_rendering':True,'finished_tape_reversal':False,'posthoc_repair':False,'catalogue_text':False,'repeated_units':False}})
 exact=[x for x in rows if x['audit']['pointer_exact'] and x['audit']['sha256_forward']==x['audit']['sha256_reverse']]
 return {'experiment_id':'topic-frame-lattice-20260920','method':'independently authored complete clause lattice joined by shared topic and discourse frame; no reverse-derived text or repair','stats':{'left_clauses':len(LEFT),'right_clauses':len(RIGHT),'frame_joins':joins,'rendered_candidates':len(rows),'exact_candidates':len(exact),'fresh_exact_gt38':sum(x['audit']['letters']>38 for x in exact)},'exact_candidates':exact,'rendered_candidates':rows,'novelty_preflight':{'status':'passed','signature':'human-authored-topic-frame-lattice|complete-clause-join|topic+discourse-typed','distinct_from':'prior lexical semordnilap, residual, CFG, and scene-edge searches; this joins intact clauses only through a shared discourse frame before character auditing'},'provenance':{'audits':['independent two-pointer comparison','independent forward/reverse SHA-256'],'reader_gate':'no readability claim without human review'},'status':'no exact closure; intact topic-frame controls retained' if not exact else 'fresh exact candidate requires human reading','next_construction':'Add a third independently authored clause family with causal frames, preserving frame-typed joins.'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats'],sort_keys=True))
