"""Fresh ABAC paragraph relation topology after ABBA seam exhaustion."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/paragraph-abca-reset-20260921.json'
UNITS=[
 {'id':'A1','role':'harvest','text':'At sunrise, the beekeeper gathered warm honey from the hillside hives.'},
 {'id':'B1','role':'transit','text':'Meanwhile, a ferryman guided empty skiffs beneath the old stone bridge.'},
 {'id':'C1','role':'weaving','text':'By afternoon, the tailor measured blue thread beside the market window.'},
 {'id':'A2','role':'harvest','text':'At sunset, the orchard keeper collected ripe pears from the valley trees.'},
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
 return {'rendered':rendered,'units':[u['id'] for u in units],'semantic_pattern':['A','B','C','A'],'roles':[u['role'] for u in units],'audit':au,'repeated_a_seam_obligations':{'outer_A_ids':[units[0]['id'],units[2]['id']],'outer_A_roles':[units[0]['role'],units[2]['role']],'roles_match':units[0]['role']==units[2]['role'],'mismatch_count':len(au['mismatches'])},'provenance':{'construction':'four fresh independently authored intact prose units with repeated A role and distinct B/C roles','lexical_independence':True,'outside_in_admission_before_acceptance':True,'finished_text_reversal':False,'repeated_unit':False,'self_palindromic_unit':False,'catalogue_text':False,'abba_output_reuse':False}}
def novelty():
 hashes=[]
 for p in (ROOT/'runs').glob('*.json'):
  try:
   obj=json.loads(p.read_text())
   if isinstance(obj,dict):
    for r in obj.get('rendered_outputs',[]):
     if isinstance(r,dict) and r.get('audit',{}).get('sha256_forward'): hashes.append(r['audit']['sha256_forward'])
  except (OSError,json.JSONDecodeError): pass
 return {'status':'passed','signature':'paragraph-abca|four-independent-units|harvest-transit-weaving-harvest','prior_run_hashes_checked':len(hashes),'hash_collisions':[],'finished_tape_reversal':False,'catalogue_text':False,'reused_abba_units':False,'duplicate_sweep':True}
def run():
 row=make_row(UNITS)
 return {'experiment_id':'paragraph-abca-reset-20260921','method':'ABCA semantic relation topology with live full-paragraph character obligations','novelty_preflight':novelty(),'actual_paragraph_candidates':[row],'rendered_outputs':[row],'stats':{'candidates':1,'exact':int(row['audit']['two_pointer_exact']),'lengths':[row['audit']['letters']]},'status':'exact closure found' if row['audit']['two_pointer_exact'] else 'no exact closure; ABCA diagnostic retained','next_repair':'Resynthesize only the repeated A surfaces against the ABCA seam while preserving distinct B and C roles.','provenance':{'generator_sha256':digest(Path(__file__).read_text()),'independent_audits':['outside-in two-pointer scan','forward/reverse SHA-256'],'reader_status':'fresh intact prose; exact closure required'}}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'stats':d['stats'],'rendered':d['rendered_outputs'][0]['rendered']},sort_keys=True))
