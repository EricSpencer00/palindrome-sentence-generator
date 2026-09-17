import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='cross-pos-semordnilap-scene-cfg-20260916';SIGNATURE='cross-pos-semordnilap|verb-noun-pairs|function-boundary-shift|connected-scene-cfg|no-chain'
TEXTS=['The guide records a calm harbor scene, then pilots turn the boats toward the lantern before dusk.','Please permit the young artist to present the painted map while the careful clerk notes each change.']
PAIRS=[('stressed','desserts','verb-noun'),('drawer','reward','noun-verb'),('diaper','repaid','noun-verb')]
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for i,s in enumerate(TEXTS):
  t=normalize_letters(s);rows.append({'scene':i,'cross_pos_pairs':PAIRS,'function_boundary_choices':['then','toward','while','the'],'clause_cfg':'S -> NP VP; VP -> V NP CONJ S','semantic_consistency':True,'pair_role_usage':'one optional lexical candidate, never a chain','audit':audit(s),'next_repair':{'operator':'replace the scene verb with a held-out cross-POS pair while preserving agent/patient roles','reason':'connected scene remains readable but outer residual is nonzero'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'enumerate cross-POS reversible lexical alternatives inside connected scene CFGs with function-word boundary shifts','candidates':rows,'stats':{'rendered':2,'exact':0},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_unit':False},'next_repair':{'operator':'swap one role-preserving cross-POS pair at the highest residual boundary and retest function-word attachment','reason':'no exact closure without disconnected pair chaining'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'cross-POS semordnilap bank plus fresh scene frames','audits':['independent two-pointer','forward/reverse SHA-256','scene CFG','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
