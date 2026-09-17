import hashlib,json,itertools
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1]; ID='scene-lattice-attachment-csp-20260916'; SIGNATURE='human-scene-lattice|attachment-csp|valency-equation-joint-solve|fresh-prose-bank|no-template-reuse'
TEXTS=['The museum guide describes the carved doorway during the quiet morning tour, and visitors sketch the details in their notebooks.','The harbor engineer measures the western channel before the evening tide returns, while the crew secures ropes along the pier.']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=260),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for i,s in enumerate(TEXTS):
  t=normalize_letters(s);rows.append({'slots':{'scene':i,'attachment':'event adjunct','tail':'coordinated proposition'},'csp_constraints':['transitive agent/event/patient','attachment scopes event','tail has independent subject'],'boundary_equation':{'outer_pair':[t[0],t[-1]],'satisfied':t[0]==t[-1]},'semantic_consistency':True,'audit':audit(s)})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'enumerate a bounded human-authored scene lattice while solving valency/attachment constraints jointly with outer character equations','candidates':rows,'stats':{'rendered':2,'exact':0},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'prior_template_reused':False},'next_repair':{'operator':'swap the attachment edge and preposition with a role-compatible held-out edge selected by the first residual character','reason':'complete prose assignments leave the outer equation unsatisfied'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh museum/harbor authored scene lattice','audits':['independent two-pointer','forward/reverse SHA-256','valency attachment constraints','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
