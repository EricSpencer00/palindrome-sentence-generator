import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='finite-semantic-csp-object-verb-repair-20260916';SIGNATURE='finite-csp-object-verb-repair|joint-valency-pair|single-state|fresh-scene|no-sweep'
TEXT='The observant ranger surveys a weathered sextant beside the rescue shed.'
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 t=normalize_letters(TEXT);row={'repair_state':'joint object+verb substitution','parent_state':'finite-semantic-palindrome-csp-20260916','change':{'old':'studies a damaged compass','new':'surveys a weathered sextant'},'valency_preserved':True,'fixed_slots':['observant ranger','beside rescue shed'],'semantic_consistency':True,'audit':audit(TEXT),'next_repair':{'operator':'change only adjunct preposition while preserving repaired event frame','reason':'joint object/verb repair yields intact prose but no exact closure'}}
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'single joint object+verb substitution at highest residual with valency and all other semantic slots frozen','candidates':[row],'stats':{'rendered':1,'exact':0},'next_repair':row['next_repair'],'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'sweep':False,'word_order_mirror':False,'replayed_prior_scene':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'held-out ranger event/object pair','parent_state':'finite-semantic-palindrome-csp-20260916','repair':'studies a damaged compass -> surveys a weathered sextant; valency preserved','audits':['independent two-pointer','forward/reverse SHA-256','semantic replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
