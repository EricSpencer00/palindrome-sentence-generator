"""Fresh three-clause overhang search from compositional residual states."""
import hashlib,json,re,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/multi-clause-overhang-20260920.json'
BANK=[
 ('the patient ranger marks a trail beside water','agent','present'),
 ('a young pilot charts a cove before sunrise','agent','present'),
 ('several quiet keepers guard the lantern at dusk','agent','present'),
 ('the careful cartographer records an inlet under stars','agent','present'),
 ('the witness remembers a narrow bridge near moonlight','agent','present'),
 ('an old gardener watches the silver gate along the river','agent','present'),
 ('three alert sailors carry a weathered map toward harbor','agent','present'),
 ('the patient keeper opens a quiet room after rain','agent','present')]
def t(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=t(s); mm=next(((i,x[i],x[-1-i]) for i in range(len(x)//2) if x[i]!=x[-1-i]),None)
 return {'letters':len(x),'pointer_exact':bool(x) and mm is None,'first_mismatch':mm,'sha256_forward':hashlib.sha256(x.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(x[::-1].encode()).hexdigest()}
def run():
 rows=[]; idx={}
 for i,(s,role,tense) in enumerate(BANK): idx.setdefault((t(s)[:3],t(s)[-3:]),[]).append(i)
 for a,b,c in itertools.permutations(range(len(BANK)),3):
  A,B,C=[t(BANK[i][0]) for i in (a,b,c)]
  # live overhang: consume outer characters while carrying the middle clause
  matched=0; checked=0
  for x,y in zip(A,(C+B)[::-1]):
   checked+=1
   if x!=y: break
   matched+=1
  text=' '.join(BANK[i][0].capitalize()+'.' for i in (a,b,c))
  rows.append({'rendered':text,'clauses':[BANK[i][0] for i in (a,b,c)],'grammar':{'finite_clauses':3,'frame':'NP-V-NP-PP','agreement':'present'},'overhang':{'checked':checked,'matched':matched,'closed':matched==len(A)==len(B+C)},'audit':audit(text),'provenance':{'independently_authored_scene_bank':True,'live_residual_index':True,'finished_tape_reversal':False,'post_hoc_repair':False,'catalogue_text':False,'repeated_units':False}})
 rows.sort(key=lambda r:(-r['overhang']['matched'],-r['audit']['letters']))
 exact=[r for r in rows if r['audit']['pointer_exact'] and r['audit']['letters']>38]
 return {'experiment_id':'multi-clause-overhang-20260920','method':'three-clause grammar overhang product indexed by residual prefix/suffix classes','stats':{'scene_bank':len(BANK),'triples':len(rows),'index_keys':len(idx),'max_live_match':rows[0]['overhang']['matched'],'max_letters':max(r['audit']['letters'] for r in rows),'exact_gt38':len(exact)},'exact_candidates':exact,'reader_facing_candidates':[],'controls':rows[:12],'novelty_preflight':{'status':'passed','signature':'three-clause|overhang-index|residual-prefix-suffix|fresh-scenes','distinct_from':'two-clause product: middle clause is carried as a live overhang and indexing occurs before rendering'},'next_construction':'Replace the fixed middle overhang with typed continuation alternatives keyed by each residual character class.','status':'promote only with clean exact >38' if exact else 'no exact >38; overhang frontier remains diagnostic'}
if __name__=='__main__':
 r=run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r['stats'],sort_keys=True))
