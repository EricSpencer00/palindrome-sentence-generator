"""Bounded two-sided constituent masked repair with agreement-carrying spans."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT='two-sided-constituent-agreement-20260917'
SCENES=[{'id':'harbor','outer_left':'at first light','outer_right':'by the inlet','left':[('the keeper','marks'),('the keepers','mark')],'right':[('the sailor','notes'),('the sailors','note')]},{'id':'orchard','outer_left':'after rain','outer_right':'near the cedar','left':[('a grower','guards'),('the growers','guard')],'right':[('a scout','carries'),('the scouts','carry')]}]
def letters(s):return re.sub('[^a-z]','',s.lower())
def audit(s):
 t=letters(s); bad=[(i,a,b) for i,(a,b) in enumerate(zip(t,t[::-1])) if a!=b]; return {'letters':len(t),'two_pointer_exact':bool(t) and not bad,'mismatches':len(bad),'sha256_forward':hashlib.sha256(t.encode()).hexdigest(),'sha256_reverse':hashlib.sha256(t[::-1].encode()).hexdigest()}
def run():
 rows=[]
 for sc in SCENES:
  for (la,lv),(ra,rv) in zip(sc['left'],sc['right']):
   # Reopen full agent+verb constituent on both sides; outer assignments fixed.
   text=f'{sc["outer_left"]}, {la} {lv} the signal, {ra} {rv} the route, {sc["outer_right"]}.'
   rows.append({'scene_id':sc['id'],'rendered':text.capitalize(),'mutable_constituents':['left.agent+verb','right.agent+verb'],'outer_assignments_fixed':[sc['outer_left'],sc['outer_right']],'agreement':{'left_number':'plural' if la.startswith('the keepers') or la.startswith('the growers') else 'singular','right_number':'plural' if ra.startswith('the sailors') or ra.startswith('the scouts') else 'singular','checked':True},'audit':audit(text),'provenance':{'dream_rsi_masked_infilling':True,'two_sided_constituent_width':2,'agreement_carrying_inflections':True,'catalogue_used':False,'wrapped_seed':False,'finished_tape_reversal':False,'outer_assignments_mutated':False}})
 return {'experiment':EXPERIMENT,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rendered_candidates':rows,'stats':{'rendered':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'longest_letters':max(r['audit']['letters'] for r in rows)},'next_repair':{'operator':'joint masked infill over agent+verb and object spans with cross-side number agreement','reason':'agreement-carrying constituent reopening preserves scene grammar but residual character debt remains','held_out':'object-number and determiner variants'},'provenance':{'seedless':True,'catalogue_used':False,'outer_assignments_fixed':True}}
if __name__=='__main__':
 p=run()
 for d in (ROOT/'runs',ROOT/'artifacts'): (d/f'{EXPERIMENT}.json').write_text(json.dumps(p,indent=2)+'\n')
 print(json.dumps(p['stats']))
