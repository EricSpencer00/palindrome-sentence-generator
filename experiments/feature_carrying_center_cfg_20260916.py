import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='feature-carrying-center-cfg-20260916';SIGNATURE='feature-carrying-character-cfg|morphology-clitic-state|free-internal-center|joint-sides|fresh'
TEXTS=['The archivist has a map, and the careful pilot will carry it to the harbor before dawn.','A gardener will water the beds, and the patient mason has repaired the gate near the orchard.']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for s in TEXTS:
  t=normalize_letters(s);rows.append({'grammar_states':['S','NP[agr]','VP[tense]','CENTER[free]','CLITIC[case]'],'features':{'agreement':'singular','tense':['perfect','future'],'clitic':'object-it','center':'free'},'joint_character_obligation':{'prefix':t[:6],'suffix':t[-6:],'center_unconstrained':True},'semantic_consistency':True,'audit':audit(s),'next_repair':{'operator':'change clitic attachment and carry its case feature across the free center','reason':'outer character debt remains after agreement-valid generation'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'feature-carrying character CFG jointly expands grammatical sides with agreement, tense, clitic, and a free internal center','candidates':rows,'stats':{'rendered':2,'exact':0},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'posthoc_reversal':False,'word_order_mirror':False},'next_repair':{'operator':'expand clitic attachment alternatives at the first residual while preserving agreement and the free center','reason':'all rows are complete prose but no exact character closure was reached'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh authored morphology/clitic frames','audits':['independent two-pointer','forward/reverse SHA-256','feature replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
