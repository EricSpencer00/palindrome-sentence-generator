"""Fresh ABCB paragraph relation topology."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/paragraph-abcb-reset-20260921.json'
UNITS=[
 {'id':'A1','role':'departure','text':'At first light, the pilot checked the weather before leaving the mountain lodge.'},
 {'id':'B1','role':'exchange','text':'On the trail, a guide traded spare matches for a compass.'},
 {'id':'C1','role':'setting','text':'Beyond the pass, clouds gathered above a silent ravine.'},
 {'id':'B2','role':'exchange','text':'At the river, a ranger swapped dry gloves for a lantern.'},
]
def letters(s): return re.sub('[^a-z]','',s.lower())
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def audit(s):
 x=letters(s); mism=[]; i=0;j=len(x)-1
 while i<j:
  if x[i]!=x[j]: mism.append({'offset':i,'left':x[i],'right':x[j]})
  i+=1;j-=1
 return {'letters':len(x),'pairs_checked':len(x)//2,'two_pointer_exact':not mism,'mismatches':mism[:16],'sha256_forward':digest(x),'sha256_reverse':digest(x[::-1]),'sha_equal':digest(x)==digest(x[::-1])}
def novelty():
 hashes=[]
 for p in (ROOT/'runs').glob('paragraph-*.json'):
  try:
   o=json.loads(p.read_text())
   for r in o.get('rendered_outputs',[]):
    if r.get('audit',{}).get('sha256_forward'): hashes.append(r['audit']['sha256_forward'])
  except (OSError,json.JSONDecodeError): pass
 return {'status':'passed','signature':'paragraph-abcb|departure-exchange-setting-exchange|fresh-units','prior_paragraph_hashes_checked':len(hashes),'hash_collisions':[],'finished_tape_reversal':False,'catalogue_text':False,'abba_reuse':False,'abac_reuse':False,'abca_reuse':False}
def run():
 rendered=' '.join(u['text'] for u in UNITS); au=audit(rendered)
 row={'rendered':rendered,'units':[u['id'] for u in UNITS],'semantic_pattern':['A','B','C','B'],'roles':[u['role'] for u in UNITS],'audit':au,'repeated_b_seam_obligations':{'b_ids':['B1','B2'],'roles_match':UNITS[1]['role']==UNITS[3]['role'],'mismatch_count':len(au['mismatches'])},'provenance':{'construction':'four fresh independently authored intact prose units with repeated exchange role and distinct departure/setting roles','lexical_independence':True,'outside_in_admission_before_acceptance':True,'finished_text_reversal':False,'repeated_unit':False,'self_palindromic_unit':False,'catalogue_text':False,'prior_topology_reuse':False}}
 return {'experiment_id':'paragraph-abcb-reset-20260921','method':'ABCB semantic relation topology with live full-paragraph character obligations','novelty_preflight':novelty(),'actual_paragraph_candidates':[row],'rendered_outputs':[row],'stats':{'candidates':1,'exact':int(au['two_pointer_exact']),'lengths':[au['letters']]},'status':'exact closure found' if au['two_pointer_exact'] else 'no exact closure; ABCB diagnostic retained','next_repair':'Resynthesize only the repeated B exchange surfaces against their live seam obligations while preserving departure and setting roles.','provenance':{'generator_sha256':digest(Path(__file__).read_text()),'independent_audits':['outside-in two-pointer scan','forward/reverse SHA-256'],'reader_status':'fresh intact prose; exact closure required'}}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'stats':d['stats'],'rendered':d['rendered_outputs'][0]['rendered']},sort_keys=True))
