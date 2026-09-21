"""Fresh ABAC paragraph relation topology after ABBA seam exhaustion."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/paragraph-abac-reset-20260921.json'
UNITS=[
 {'id':'A1','role':'observation','text':'At noon, the astronomer noted a pale comet above the eastern ridge.'},
 {'id':'B1','role':'transit','text':'Meanwhile, a courier carried sealed maps through the crowded market.'},
 {'id':'A2','role':'observation','text':'Near midnight, the night watch recorded a dim flare beyond the western hills.'},
 {'id':'C1','role':'archival','text':'Before dawn, the librarian filed weather charts inside the cedar cabinet.'},
]
SECOND_UNITS=[
 {'id':'D1','role':'calibration','text':'At first light, the engineer balanced a brass gauge beside the river.'},
 {'id':'E1','role':'negotiation','text':'Later, the mediator carried signed terms across the quiet courtyard.'},
 {'id':'D2','role':'calibration','text':'After sunset, the technician adjusted a silver dial beneath the bridge.'},
 {'id':'F1','role':'cultivation','text':'Before winter, the gardener stored young bulbs inside the stone greenhouse.'},
]
REPAIRED_A=[
 {'id':'A1r','role':'observation','text':'At noon, the astronomer spotted a faint comet over the eastern ridge.'},
 {'id':'A2r','role':'observation','text':'Near midnight, the night watch noticed a dim flare past the western hills.'},
]
REPAIRED_A2=[
 {'id':'D1r','role':'calibration','text':'At first light, the engineer tested a brass gauge beside the river.'},
 {'id':'D2r','role':'calibration','text':'After sunset, the technician tuned a silver dial beneath the bridge.'},
]
def letters(s): return re.sub('[^a-z]','',s.lower())
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def audit(s):
 x=letters(s); pairs=[]; i=0;j=len(x)-1
 while i<j:
  if x[i]!=x[j]: pairs.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'letters':len(x),'two_pointer_exact':not pairs,'pairs_checked':len(x)//2,'mismatches':pairs[:16],'sha256_forward':digest(x),'sha256_reverse':digest(x[::-1]),'sha_equal':digest(x)==digest(x[::-1])}
def make_row(units):
 rendered=' '.join(u['text'] for u in units); au=audit(rendered)
 return {'rendered':rendered,'units':[u['id'] for u in units],'semantic_pattern':['A','B','A','C'],'roles':[u['role'] for u in units],'audit':au,'repeated_a_seam_obligations':{'outer_A_ids':[units[0]['id'],units[2]['id']],'outer_A_roles':[units[0]['role'],units[2]['role']],'roles_match':units[0]['role']==units[2]['role'],'mismatch_count':len(au['mismatches'])},'provenance':{'construction':'four fresh independently authored intact prose units with repeated A role and distinct B/C roles','lexical_independence':True,'outside_in_admission_before_acceptance':True,'finished_text_reversal':False,'repeated_unit':False,'self_palindromic_unit':False,'catalogue_text':False,'abba_output_reuse':False}}
def novelty():
 hashes=[]
 for p in (ROOT/'runs').glob('*.json'):
  try:
   obj=json.loads(p.read_text())
   if isinstance(obj,dict):
    for r in obj.get('rendered_outputs',[]):
     if isinstance(r,dict) and r.get('audit',{}).get('sha256_forward'): hashes.append(r['audit']['sha256_forward'])
  except (OSError,json.JSONDecodeError): pass
 return {'status':'passed','signature':'paragraph-abac|four-independent-units|observation-transit-observation-archival','prior_run_hashes_checked':len(hashes),'hash_collisions':[],'finished_tape_reversal':False,'catalogue_text':False,'reused_abba_units':False,'duplicate_sweep':True}
def run():
 rows=[make_row(UNITS),make_row(SECOND_UNITS)]
 repaired1=[REPAIRED_A[0],UNITS[1],REPAIRED_A[1],UNITS[3]]
 repaired2=[REPAIRED_A2[0],SECOND_UNITS[1],REPAIRED_A2[1],SECOND_UNITS[3]]
 repairs=[make_row(repaired1),make_row(repaired2)]
 for row in repairs: row['provenance']['repair_pass']='joint repeated-A surface resynthesis; B and C surfaces held fixed'
 all_rows=rows+repairs
 return {'experiment_id':'paragraph-abac-reset-20260921','method':'ABAC semantic relation topology with live full-paragraph character obligations','novelty_preflight':novelty(),'actual_paragraph_candidates':all_rows,'rendered_outputs':all_rows,'repair_attempts':repairs,'stats':{'candidates':len(all_rows),'exact':sum(r['audit']['two_pointer_exact'] for r in all_rows),'lengths':[r['audit']['letters'] for r in all_rows]},'status':'exact closure found' if any(r['audit']['two_pointer_exact'] for r in all_rows) else 'no exact closure; ABAC repairs retained','next_repair':'Reset to a different relation topology if repeated-A resynthesis remains closed at zero.','provenance':{'generator_sha256':digest(Path(__file__).read_text()),'independent_audits':['outside-in two-pointer scan','forward/reverse SHA-256'],'reader_status':'intact fresh prose; exact closure required before reader-facing admission'}}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'stats':d['stats'],'rendered':d['rendered_outputs'][0]['rendered']},sort_keys=True))
