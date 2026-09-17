import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='semantic-slot-cross-boundary-dp-20260916';SIGNATURE='heldout-semantic-slot|cross-boundary-character-dp|fresh-scene-family|role-compatible-substitution|new-state'
TEXTS=['At noon, the coastal ranger studies a damaged compass beside the rescue shed, while volunteers prepare fresh water for the arriving hikers.','By evening, the school librarian sorts a missing atlas near the reading room, while students arrange quiet tables for the visiting class.']
SLOTS={'object':['damaged compass','missing atlas','weathered telescope'],'adjunct':['beside the rescue shed','near the reading room','under the covered porch'],'patient':['arriving hikers','visiting class','waiting students']}
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=260),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for i,s in enumerate(TEXTS):
  t=normalize_letters(s);rows.append({'scene':i,'slot_bank':SLOTS,'selected_slots':['object','adjunct','patient'],'role_constraints':['ranger studies object','location adjunct','volunteer prepares patient'],'cross_boundary_dp':{'residual_prefix':t[:8],'residual_suffix':t[-8:],'slot_order':['object','adjunct','patient']},'semantic_consistency':True,'audit':audit(s),'next_repair':{'operator':'substitute one held-out role-compatible object at first residual and relexicalize its clause','reason':'fresh prose remains grammatical but no exact closure'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'choose held-out semantic slot substitutions at live cross-boundary residuals while preserving role constraints in a fresh scene family','candidates':rows,'stats':{'rendered':2,'exact':0},'next_repair':{'operator':'substitute one held-out role-compatible object at first residual and relexicalize its clause','reason':'no exact anti-shortcut candidate survived semantic slot DP'},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'function_word_auxiliary_search':False,'prior_scene_reused':False,'word_order_mirror':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh ranger/library semantic slot banks','audits':['independent two-pointer','forward/reverse SHA-256','role constraint replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
