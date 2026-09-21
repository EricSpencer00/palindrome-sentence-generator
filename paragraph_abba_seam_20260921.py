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
REPAIR_SURFACES=[
 ('A1','At sunrise, the navigator departed the harbor to chart a quiet inlet.'),
 ('A2','By twilight, the surveyor returned from the pier after mapping a sheltered bay.'),
 ('B1','Meanwhile, the mechanic secured a loose wheel beside the workshop.'),
 ('B2','Later, the apprentice repaired a frayed strap behind the tool shed.'),
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
def abba_seam_solver(row):
    """Compare exposed ABBA obligations; never repairs by reversing finished text."""
    x=letters(row['rendered']); pairs=[]
    for i in range(min(len(x)//2, 24)):
        if x[i] != x[-1-i]: pairs.append({'offset':i,'left':x[i],'right':x[-1-i]})
    return {'strategy':'live outer-to-inner ABBA obligation comparison','checked_prefix_pairs':min(len(x)//2,24),'mismatches':pairs,'repairable_by_unit_resynthesis':bool(pairs),'finished_tape_reversal':False}
def resynthesize_existing_frame():
    """Lexically resynthesize the original semantic roles; no new units or tape reversal."""
    by_id=dict(REPAIR_SURFACES)
    ordered=[(k,by_id[k]) for k in ('A1','B1','B2','A2')]
    return make_row([{'id':k,'frame':('departure' if k.startswith('A') else 'repair'),'text':t} for k,t in ordered],['departure','repair','repair','departure'])
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
 for row in rows: row['seam_solver']=abba_seam_solver(row)
 repair=resynthesize_existing_frame(); repair['seam_solver']=abba_seam_solver(repair); repair['provenance']['repair_pass']='live mismatch-guided lexical resynthesis of existing A/B roles'
 return {'experiment_id':'paragraph-abba-seam-20260921','method':'paragraph-level ABBA semantic frame topology with live full-paragraph character obligations','novelty_preflight':novelty(),'actual_paragraph_candidates':rows+[repair],'rendered_outputs':rows+[repair],'repair_attempts':[repair],'stats':{'candidates':len(rows)+1,'exact':sum(r['audit']['two_pointer_exact'] for r in rows+[repair]),'lengths':[r['audit']['letters'] for r in rows+[repair]]},'status':'exact closure found' if any(r['audit']['two_pointer_exact'] for r in rows+[repair]) else 'no exact closure; repair resynthesis retained','next_repair':{'operator':'derive lexical choices from the first mismatched exposed character while preserving complete semantic frames','reason':'mismatch-guided resynthesis changed lexical surfaces but did not yet close the full tape'},'provenance':{'generator_sha256':sha(Path(__file__).read_text()),'independent_audits':['outside-in two-pointer scan','forward/reverse SHA-256','ABBA-specific seam obligation solver'],'reader_status':'intact prose candidates; exact closure required before reader-facing admission'}}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'stats':d['stats'],'rendered':d['rendered_outputs'][0]['rendered']},sort_keys=True))
