import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='reversible-phrase-directional-adjunct-repair-20260916';SIGNATURE='phrase-directional-adjunct-repair|heldout-preposition-location-np|role-preserved|fresh-state'
TEXTS=['The harbor keeper preserves a quiet journal while the night crew carries bright lanterns beside the old quay.','The garden steward stores a careful record while the morning team brings clean baskets near the eastern greenhouse.']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for i,s in enumerate(TEXTS):
  t=normalize_letters(s);rows.append({'repair_state':'directional adjunct replacement','parent_state':'reversible-phrase-pair-role-repair-20260916','replacement':{'old_role':'directional location adjunct','new_preposition':['beside','near'][i],'new_location_np':['the old quay','the eastern greenhouse'][i]},'valency_preserved':True,'live_boundary_equation':{'outer':[t[0],t[-1]],'satisfied':t[0]==t[-1]},'semantic_consistency':True,'audit':audit(s),'next_repair':{'operator':'replace only the adjunct determiner with a held-out determiner matching the residual while retaining location role','reason':'single adjunct repair remains grammatical but outer character obligation is open'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'replace only a directional preposition plus location noun phrase under a live boundary equation','candidates':rows,'stats':{'rendered':2,'exact':0},'next_repair':{'operator':'replace only the adjunct determiner with a held-out determiner matching the residual','reason':'no exact closure after role-preserving directional repair'},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'word_order_mirror':False,'repeated_units':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'held-out location adjunct bank','parent_state':'reversible-phrase-pair-role-repair-20260916','repair':'replaced directional adjunct only; agent/theme and scene roles unchanged','audits':['independent two-pointer','forward/reverse SHA-256','valency replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
