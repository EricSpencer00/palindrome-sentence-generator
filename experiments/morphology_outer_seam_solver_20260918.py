"""Feature-carrying outer determiner/name seam solver (fresh repair)."""
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; NAME="morphology-outer-seam-solver-20260918"
LEX={"sg":{"det":["a","the"],"subj":["baker","pilot","clerk"],"verb":["marks","opens","guards"]},"pl":{"det":["some","the"],"subj":["bakers","pilots","clerks"],"verb":["mark","open","guard"]}}
def letters(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); r=t[::-1]; exact=bool(t) and all(a==b for a,b in zip(t,r)); return {'letters':len(t),'two_pointer_exact':exact,'mismatches':sum(a!=b for a,b in zip(t,r)),'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(r.encode()).hexdigest()}
def seam_ok(left,right):
 a,b=letters(left),letters(right); return all(x==y for x,y in zip(a[:min(len(a),len(b))],b[::-1][:min(len(a),len(b))]))
def run():
 states=[]
 # Each transition carries number and tense; right grows in reverse role order.
 for num in LEX:
  for det in LEX[num]['det']:
   for subj in LEX[num]['subj']:
    for verb in LEX[num]['verb']:
     left=f'{det} {subj} {verb}'
     for rverb in LEX[num]['verb']:
      for rsubj in LEX[num]['subj']:
       right=f'{rverb} {rsubj} {det}'
       if seam_ok(left,right): states.append((left,right,num,'present'))
 rows=[{'candidate_id':f'mos-{i}','rendered':f'{l}; {r}.','audit':audit(f'{l}; {r}.'),'reader_status':'human-unreviewed','provenance':{'fresh_authored_lexicon':True,'catalogue_used':False,'finished_tape_reversal':False,'repeated_self_palindromic_unit':False,'features':{'number':n,'tense':t}}} for i,(l,r,n,t) in enumerate(states[:8])]
 controls=['The baker marks maps; a pilot opens doors.','Some clerks guard gates; the pilots mark maps.']
 return {'experiment':NAME,'method':'agreement-carrying morphology with reverse-residual outer seam','construction':{'repair_operator':'joint number/tense transition plus reverse role seam','live_character_constraints_before_render':True,'independent_audit':'two_pointer_and_sha256'},'rendered_candidates':rows,'rendered_controls':[{'rendered':x,'audit':audit(x),'provenance':{'fresh_authored_control':True}} for x in controls],'stats':{'states':len(states),'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max([x['audit']['letters'] for x in rows] or [0])},'novelty_preflight':{'new_geometry':'feature-carrying reverse residual transitions','prior_lane_reused':False,'duplicate_sweep':False},'reader_gate':{'status':'not_triggered','programmatic_metrics_are_diagnostic':True},'next_repair':{'operator':'learn compatible inflectional seam tokens from authored clause pairs','reason':'determiner/name shell remains unclosed'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'human_readability_certified':False}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'):
  d.mkdir(exist_ok=True); (d/f'{NAME}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))
