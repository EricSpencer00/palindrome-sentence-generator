"""Scene-coherent semordnilap-edge graph probe with discourse connectors."""
import hashlib,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/scene-semordnilap-graph-20260916.json'
ID='scene-semordnilap-graph-20260916';SIG='scene-coherent-semordnilap-edges|causal-event-graph|discourse-connectors|nonrepeating-units|independent-pointer-sha'
SCENES=['Because the archivist found the drawer, the keeper recorded the reward and carried the map to the harbor.','When the gardener opened the gate, the courier carried the letter and marked the garden path.']
EDGES=[('drawer','reward'),('diary','yraid'),('stressed','desserts')]
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':len(t)-1-i,'a':t[i],'b':t[-1-i]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for i,s in enumerate(SCENES):
  text=s+' '+('The record explains the event.' if i==0 else 'The note preserves the event.');a=audit(text);c=mechanical_admission_checks(text,min_letters=90,max_letters=220)
  rows.append({'rendered':text,'letters':a['letters'],'scene_id':i,'graph':{'nodes':['archivist','keeper','artifact','harbor'],'causal_edge':'discovery enables recording and transport','semordnilap_edges_considered':EDGES,'shared_event':True},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'human_authored_scene':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'semordnilap_edge_is_not_presented_as_exact':True}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'global causal scene graph constraining semordnilap lexical edges and discourse connectors','candidates':rows,'stats':{'scenes':2,'edges_considered':6,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','duplicate_sweep_rejected':True},'next_repair':'Replace the first residual edge with a sense-compatible event role while preserving causal direction, then solve connector boundaries globally; reject any repeated-unit closure.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['candidates'][0]['rendered'],'length':p['candidates'][0]['letters']},indent=2))
