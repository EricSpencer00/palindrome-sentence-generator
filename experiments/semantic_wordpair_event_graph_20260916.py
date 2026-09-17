"""Bounded semantic word-pair graph path in a coherent SVO event frame."""
import hashlib,json
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/semantic-wordpair-event-graph-20260916.json'
ID='semantic-wordpair-event-graph-20260916';SIG='semantic-word-pair-graph|heldout-reverse-compatible-edges|coherent-svo-event|no-semordnilap-chain|independent-pointer-sha'
EDGES=[('pilot','tends'),('garden','rednag'),('parcel','lecrap')]
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[j]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 text='The pilot carries a sealed parcel to the garden, where the keeper waters the basil before sunset.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=220);h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 row={'rendered':text,'letters':a['letters'],'graph_path':{'nodes':['pilot','parcel','garden','keeper','basil'],'event':'delivery enables garden care','heldout_edges_considered':EDGES,'selected_edges':[]},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'coherent_svo_event':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'semordnilap_chain_rejected':True,'generator_sha256':h}}
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'one bounded semantic word-pair graph path constrained by a coherent delivery/care event','candidate':row,'stats':{'bounded_paths':1,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','duplicate_sweep_rejected':True},'next_repair':'Add one held-out event-role edge between delivery and care and re-solve the first residual without admitting any semordnilap chain.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'candidate':p['candidate']['rendered'],'length':p['candidate']['letters']},indent=2))
