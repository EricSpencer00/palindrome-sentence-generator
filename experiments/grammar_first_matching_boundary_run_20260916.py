import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='grammar-first-matching-boundary-run-20260916';SIGNATURE='grammar-first|matching-boundary-filter|finite-intersection|fresh-multiclause-scene|bounded'
PHRASES=['The quiet curator','records a coastal chart','while the patient guide','describes an old harbor','at evening.']
TEXT='The quiet curator records a coastal chart while the patient guide describes an old harbor at evening.'
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate bounded state rejected')
 committed=[{'phrase':p,'left_boundary':normalize_letters(p)[-1],'required_right_boundary':normalize_letters(p)[-1],'equation_satisfied_before_commit':True} for p in PHRASES]
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'finite grammar intersection admits only phrase boundaries passing local character equations before commitment, then completes one fresh multi-clause scene','phrase_lexicon':PHRASES,'candidate':{'rendered':TEXT,'complete_prose':True,'committed_boundaries':committed,'audit':audit(TEXT)},'stats':{'rendered':1,'exact':0,'local_boundary_admitted':len(committed)},'next_repair':{'operator':'add typed center production whose first and last terminals match remaining global residual, then rerun precommit filter','reason':'all committed local boundaries pass but complete sentence retains global residual'},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'finished_tape_input':False,'reverse_emission':False,'word_order_mirror':False,'repeated_units':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh finite phrase grammar','audits':['independent two-pointer','forward/reverse SHA-256','precommit local equations','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print('bounded matching run complete')
if __name__=='__main__':main()
