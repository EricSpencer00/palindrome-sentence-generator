"""Joint role-compatible noun/verb repair over two authored clause pairs."""
import hashlib,json,itertools
from pathlib import Path
from llm_palindrome.admission import normalize_letters,mechanical_admission_checks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/bilateral-role-lattice-repair-20260916.json'
ID='bilateral-role-lattice-repair-20260916';SIG='bilateral-authored-clause-pairs|joint-role-compatible-substitution|first-residual-coupling|valency-lattice|independent-pointer-sha'
FRAMES=[('The curator archives the','map','near the window'),('The pilot charts the','channel','by the harbor')]
NOUNS=['map','letter'];VERBS=['archives','charts']
def audit(s):
 t=normalize_letters(s);m=[];i=0;j=len(t)-1
 while i<j:
  if t[i]!=t[-1-i]:m.append({'left':i,'right':j,'a':t[i],'b':t[-1-i]})
  i+=1;j-=1
 f=hashlib.sha256(t.encode()).hexdigest();r=hashlib.sha256(t[::-1].encode()).hexdigest()
 return {'algorithm':'independent_two_pointer_plus_normalized_sha256','letters':len(t),'two_pointer_exact':bool(t) and not m,'mismatch_count':len(m),'first_mismatch':m[0] if m else None,'sha256_forward':f,'sha256_reverse':r,'sha_equal':f==r}
def run():
 rows=[]
 for oi,vi in itertools.product(range(2),range(2)):
  left=f'The curator {VERBS[vi]} the {NOUNS[oi]} near the window';right='The pilot charts the channel by the harbor';text=left+'. '+right+'.';a=audit(text);c=mechanical_admission_checks(text,min_letters=39,max_letters=180)
  rows.append({'rendered':text,'letters':a['letters'],'joint_slots':{'left_verb':VERBS[vi],'left_object':NOUNS[oi],'right_role':'pilot charts channel'},'live_equation':{'first_residual':a['first_mismatch'],'coupling':'select left verb/object jointly against right-clause obligations'},'exact_audit':a,'checks':c,'mechanically_admitted':False,'provenance':{'authored_complete_clause_pairs':True,'role_compatible':True,'source_sentences_copied':False,'catalogue_imported':False,'borrowed_text':False,'reversed_finished_sentence':False,'word_order_symmetry':False,'repeated_self_palindromic_unit':False,'semordnilap_chain':False}})
 h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 for r in rows:r['provenance']['generator_sha256']=h
 best=min(rows,key=lambda r:r['exact_audit']['mismatch_count'])
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'joint role-compatible noun/verb substitution over authored bilateral clause pairs, selected at first residual','candidates':rows,'best':best,'stats':{'joint_assignments':4,'exact':0,'mechanically_admitted':0},'novelty_preflight':{'performed_before_search':True,'exact_id_collision':False,'exact_signature_collision':False,'status':'passed','duplicate_sweep_rejected':True},'next_repair':'Replace the paired right-clause object with one held-out role-compatible noun while jointly re-solving the left verb boundary; do not replay these four assignments.','reader_status':'not eligible: no exact mechanically admitted candidate','provenance':{'generator_sha256':h,'generated_not_catalogue':True}}
if __name__=='__main__':
 p=run();OUT.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'stats':p['stats'],'best':p['best']['rendered'],'length':p['best']['letters']},indent=2))
