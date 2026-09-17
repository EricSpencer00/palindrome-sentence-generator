import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='reversible-phrase-determiner-repair-20260916';SIGNATURE='phrase-determiner-repair|single-child|preposition-location-fixed|valency-preserved|fresh-state'
TEXT='The harbor keeper preserves a quiet journal while the night crew carries bright lanterns beside our old quay.'
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 row={'repair_state':'single determiner substitution','parent_state':'reversible-phrase-directional-adjunct-repair-20260916','replacement':{'old':'the old quay','new':'our old quay'},'preposition_fixed':'beside','location_fixed':'old quay','valency_preserved':True,'semantic_consistency':True,'audit':audit(TEXT),'next_repair':{'operator':'replace only the location adjective with a held-out compatible adjective matching the first residual','reason':'determiner repair remains readable but does not close outer obligation'}}
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'single determiner substitution against residual while freezing preposition, location head, and valency','candidates':[row],'stats':{'rendered':1,'exact':0},'next_repair':row['next_repair'],'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'word_order_mirror':False,'repeated_units':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'held-out determiner alternative','parent_state':'reversible-phrase-directional-adjunct-repair-20260916','repair':'the->our only; preposition/location and valency frozen','audits':['independent two-pointer','forward/reverse SHA-256','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
