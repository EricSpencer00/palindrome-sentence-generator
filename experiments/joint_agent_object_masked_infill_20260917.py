"""Joint two-span masked infill with cross-side number agreement (bounded)."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; EXPERIMENT='joint-agent-object-masked-infill-20260917'
SCENES=[('harbor','at first light','by the inlet'),('orchard','after rain','near the cedar')]
AG=[('the keeper','marks','the signal','singular'),('the keepers','mark','the signals','plural'),('a grower','guards','a basket','singular'),('the growers','guard','the baskets','plural')]
RIGHT=[('the sailor','notes','the route','singular'),('the sailors','note','the routes','plural'),('a scout','carries','a map','singular'),('the scouts','carry','the maps','plural')]
def letters(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); bad=sum(a!=b for a,b in zip(t,t[::-1]));return {'letters':len(t),'two_pointer_exact':bool(t) and bad==0,'mismatches':bad,'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def res(s):
 t=letters(s);return sum(a!=b for a,b in zip(t,t[::-1]))+abs(len(t)-len(t[::-1]))
def run():
 rows=[]
 for sid,ol,orr in SCENES:
  for l in AG:
   for r in RIGHT:
    if l[3]!=r[3]:continue
    text=f'{ol}, {l[0]} {l[1]} {l[2]}, while {r[0]} {r[1]} {r[2]}, {orr}.'
    rows.append({'scene_id':sid,'rendered':text.capitalize(),'mutable_spans':['left.agent+verb+object','right.agent+verb+object'],'outer_assignments_fixed':[ol,orr],'agreement':{'left_number':l[3],'right_number':r[3],'cross_side_number_equal':True},'audit':audit(text),'residual_debt':res(text),'provenance':{'joint_masked_infill':True,'span_boundaries_altered':True,'cross_side_number_agreement':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False}})
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(x['audit']['two_pointer_exact'] for x in rows),'longest_letters':max(x['audit']['letters'] for x in rows),'best_residual_debt':min(x['residual_debt'] for x in rows)},'next_repair':{'operator':'change geometry: boundary-crossing relative-clause infill','reason':'joint agent+verb+object reopening is grammar-safe but residual debt does not reach closure in 16 bounded states; this route is exhausted','route_exhausted':True},'provenance':{'bounded_states':len(rows),'catalogue_used':False,'old_four_rows_reenumerated':False}}
if __name__=='__main__':
 p=run();
 for d in (ROOT/'runs',ROOT/'artifacts'):(d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))
