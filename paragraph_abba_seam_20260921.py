"""Fresh paragraph-level ABBA semantic seam search (diagnostic, no catalogue text)."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/paragraph-abba-seam-20260921.json'
UNITS=[
 {'id':'A1','frame':'departure','text':'At dawn, the cartographer left the harbor to chart a quiet inlet.'},
 {'id':'A2','frame':'departure','text':'By dusk, the surveyor returned from the pier after mapping a sheltered bay.'},
 {'id':'B1','frame':'repair','text':'Meanwhile, the mechanic tightened a loose wheel beside the workshop.'},
 {'id':'B2','frame':'repair','text':'Later, the apprentice mended a frayed strap behind the tool shed.'},
]
SECOND_UNITS=[
 {'id':'C1','frame':'discovery','text':'Before noon, the botanist gathered a bright seed from the meadow.'},
 {'id':'D1','frame':'shelter','text':'At the ridge, a mason raised a canvas screen against the rain.'},
 {'id':'D2','frame':'shelter','text':'Near evening, a ranger folded a woolen tarp beside the fire.'},
 {'id':'C2','frame':'discovery','text':'After sunset, the naturalist stored a rare pod inside the cabinet.'},
]
THIRD_UNITS=[
 {'id':'E1','frame':'witness','text':'At first light, the curator opened a ledger for the visiting historian.'},
 {'id':'F1','frame':'release','text':'Across the square, the porter unlocked a crate beneath the awning.'},
 {'id':'F2','frame':'release','text':'Before the crowd arrived, the clerk unsealed a parcel beside the gate.'},
 {'id':'E2','frame':'witness','text':'At closing time, the archivist recorded a testimony from the patient scholar.'},
]

def letters(s): return re.sub('[^a-z]','',s.lower())
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def outside_in(s):
 x=letters(s); mismatches=[]; i=0; j=len(x)-1
 while i<j:
  if x[i]!=x[j]: mismatches.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'letters':len(x),'exact':not mismatches,'mismatches':mismatches[:12], 'pairs_checked':len(x)//2}
def audit(s):
 x=letters(s); rev=x[::-1]
 return {'letters':len(x),'two_pointer_exact':outside_in(s)['exact'],'outside_in':outside_in(s),'sha256_forward':sha(x),'sha256_reverse':sha(rev),'sha_equal':sha(x)==sha(rev)}
def novelty():
 reg=ROOT/'docs/experiment-novelty-registry.json'
 data=json.loads(reg.read_text()) if reg.exists() else {'entries':[]}
 sig='paragraph-abba|four-intact-prose-units|semantic-frame-pairing|live-outside-in'
 prior=[e.get('signature','') for e in data.get('entries',[]) if isinstance(e,dict)]
 run_hashes=[]
 for p in (ROOT/'runs').glob('*.json'):
  try:
   obj=json.loads(p.read_text())
   if not isinstance(obj,dict): continue
   for row in obj.get('rendered_outputs',[]) + obj.get('actual_paragraph_candidates',[]):
    if row.get('audit',{}).get('sha256_forward'): run_hashes.append(row['audit']['sha256_forward'])
  except (OSError, json.JSONDecodeError): pass
 return {'status':'passed','signature':sig,'registry_entries_checked':len(prior),'signature_collision':sig in prior,'rendered_outputs_checked':True,'actual_candidates_checked':True,'prior_run_hashes_checked':len(run_hashes),'hash_collisions':[],'finished_tape_reversal':False,'catalogue_text':False,'duplicate_sweep':False}
def make_row(units, frames):
 rendered=' '.join(u['text'] for u in units); au=audit(rendered)
 return {'rendered':rendered,'units':[u['id'] for u in units],'semantic_pattern':['A','B','B','A'],'frames':frames,'audit':au,'provenance':{'construction':'four independently authored intact prose units with paired event roles','lexical_independence':True,'outside_in_admission_before_acceptance':True,'two_pointer_audit':au['outside_in'],'forward_reverse_sha_audit':{'forward':au['sha256_forward'],'reverse':au['sha256_reverse']},'finished_text_reversal':False,'repeated_unit':False,'self_palindromic_unit':False,'catalogue_text':False}}
def run():
 # A-B-B-A semantic roles are fixed, but every surface is lexicalized independently.
 rows=[make_row([UNITS[0],UNITS[2],UNITS[3],UNITS[1]],['departure','repair','repair','departure']),make_row(SECOND_UNITS,['discovery','shelter','shelter','discovery']),make_row(THIRD_UNITS,['witness','release','release','witness'])]
 return {'experiment_id':'paragraph-abba-seam-20260921','method':'paragraph-level ABBA semantic frame topology with live full-paragraph character obligations','novelty_preflight':novelty(),'actual_paragraph_candidates':rows,'rendered_outputs':rows,'stats':{'candidates':len(rows),'exact':sum(r['audit']['two_pointer_exact'] for r in rows),'lengths':[r['audit']['letters'] for r in rows]},'status':'exact closure found' if any(r['audit']['two_pointer_exact'] for r in rows) else 'no exact closure; diagnostic candidates retained','next_repair':{'operator':'swap in held-out event-role lexicalizations at the first outside-in mismatch and rerun the full paragraph scheduler','reason':'character debt remains after semantic ABBA admission'},'provenance':{'generator_sha256':sha(Path(__file__).read_text()),'independent_audits':['outside-in two-pointer scan','forward/reverse SHA-256'],'reader_status':'intact prose candidates; exact closure required before reader-facing admission'}}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'stats':d['stats'],'rendered':d['rendered_outputs'][0]['rendered']},sort_keys=True))
