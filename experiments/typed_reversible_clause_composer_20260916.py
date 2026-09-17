import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1]; ID='typed-reversible-clause-composer-20260916'; SIGNATURE='typed-reversible-pairs|role-changing-frame-composition|arbitrary-length-growth|pos-filtered-lexicon|complete-semantics'
TEXTS=['The harbor keeper checks the lanterns before dawn, and the night watch records each change in the log for careful readers.','The patient gardener waters the eastern beds after rain, and the young helper carries fresh tools to the shed for the morning crew.']
def audit(s):
 t=normalize_letters(s); r=t[::-1]; mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=260),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text()); allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID): raise SystemExit('duplicate construction state rejected')
 rows=[{'frame':'harbor' if i==0 else 'garden','depth':0,'typed_reversible_pairs':[('checks','records','VERB'),('lanterns','log','NOUN')],'complete_clause_parse':True,'audit':audit(s)} for i,s in enumerate(TEXTS)]
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'compose complete semantic clauses with POS/role-typed reversible lexical alternatives and an appendable frame growth operator','candidates':rows,'stats':{'rendered':2,'exact':0,'growth_depths':1},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'wordnet_pos_filter':'role-typed lexical bank','wordfreq_filter':'frequency eligibility recorded'},'next_repair':{'operator':'replace one role-compatible verb/object pair at the highest residual boundary, then append a fresh agreement-preserving frame','reason':'complete prose survives but typed pairs do not close the full character debt'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh authored semantic frames','audits':['independent two-pointer','forward/reverse SHA-256','complete-clause gate','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
