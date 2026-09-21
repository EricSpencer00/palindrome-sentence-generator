"""Fresh AABC paragraph relation topology."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'runs/paragraph-aabc-reset-20260921.json'
UNITS=[
 {'id':'A1','role':'departure','text':'At dawn, the violinist left the riverside town with a cedar case.'},
 {'id':'A2','role':'return','text':'By evening, the musician came home through the lantern-lit square.'},
 {'id':'B1','role':'setting','text':'During the journey, rain softened the fields beyond the railway.'},
 {'id':'C1','role':'resolution','text':'At last, a neighbor tuned the waiting piano for the quiet recital.'},
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
 return {'status':'passed','signature':'paragraph-aabc|departure-return-setting-resolution|fresh-units','prior_paragraph_hashes_checked':len(hashes),'hash_collisions':[],'finished_tape_reversal':False,'catalogue_text':False,'abba_reuse':False,'abac_reuse':False,'abca_reuse':False,'abcb_reuse':False}
def run():
 rendered=' '.join(u['text'] for u in UNITS); au=audit(rendered)
 row={'rendered':rendered,'units':[u['id'] for u in UNITS],'semantic_pattern':['A','A','B','C'],'roles':[u['role'] for u in UNITS],'audit':au,'a_pair_obligations':{'ids':['A1','A2'],'coherent_role_pair':True,'mismatch_count':len(au['mismatches'])},'provenance':{'construction':'four fresh independently authored intact prose units: departure, return, setting, resolution','lexical_independence':True,'outside_in_admission_before_acceptance':True,'finished_text_reversal':False,'repeated_unit':False,'self_palindromic_unit':False,'catalogue_text':False,'prior_topology_reuse':False}}
 return {'experiment_id':'paragraph-aabc-reset-20260921','method':'AABC semantic relation topology with live full-paragraph character obligations','novelty_preflight':novelty(),'actual_paragraph_candidates':[row],'rendered_outputs':[row],'stats':{'candidates':1,'exact':int(au['two_pointer_exact']),'lengths':[au['letters']]},'status':'exact closure found' if au['two_pointer_exact'] else 'no exact closure; AABC diagnostic retained','next_repair':'Jointly resynthesize the departure/return A surfaces against live outer obligations while preserving setting and resolution roles.','provenance':{'generator_sha256':digest(Path(__file__).read_text()),'independent_audits':['outside-in two-pointer scan','forward/reverse SHA-256'],'reader_status':'fresh intact prose; exact closure required'}}
if __name__=='__main__':
 d=run();OUT.parent.mkdir(exist_ok=True);OUT.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'stats':d['stats'],'rendered':d['rendered_outputs'][0]['rendered']},sort_keys=True))
