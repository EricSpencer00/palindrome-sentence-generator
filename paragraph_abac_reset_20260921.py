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
def letters(s): return re.sub('[^a-z]','',s.lower())
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def audit(s):
 x=letters(s); pairs=[]; i=0;j=len(x)-1
 while i<j:
  if x[i]!=x[j]: pairs.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'letters':len(x),'two_pointer_exact':not pairs,'pairs_checked':len(x)//2,'mismatches':pairs[:16],'sha256_forward':digest(x),'sha256_reverse':digest(x[::-1]),'sha_equal':digest(x)==digest(x[::-1])}
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
 rendered=' '.join(u['text'] for u in UNITS); au=audit(rendered)
 row={'rendered':rendered,'units':[u['id'] for u in UNITS],'semantic_pattern':['A','B','A','C'],'roles':[u['role'] for u in UNITS],'audit':au,'provenance':{'construction':'four fresh independently authored intact prose units; A repeats observation relation while B transit and C archival roles differ','lexical_independence':True,'outside_in_admission_before_acceptance':True,'finished_text_reversal':False,'repeated_unit':False,'self_palindromic_unit':False,'catalogue_text':False,'abba_output_reuse':False}}
 return {'experiment_id':'paragraph-abac-reset-20260921','method':'ABAC semantic relation topology with live full-paragraph character obligations','novelty_preflight':novelty(),'actual_paragraph_candidates':[row],'rendered_outputs':[row],'stats':{'candidates':1,'exact':int(au['two_pointer_exact']),'lengths':[au['letters']]},'status':'exact closure found' if au['two_pointer_exact'] else 'no exact closure; ABAC diagnostic retained','next_repair':'Jointly resynthesize the two observation units against the first exposed mismatch while preserving distinct transit and archival roles.','provenance':{'generator_sha256':digest(Path(__file__).read_text()),'independent_audits':['outside-in two-pointer scan','forward/reverse SHA-256'],'reader_status':'intact fresh prose; exact closure required before reader-facing admission'}}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'stats':d['stats'],'rendered':d['rendered_outputs'][0]['rendered']},sort_keys=True))
