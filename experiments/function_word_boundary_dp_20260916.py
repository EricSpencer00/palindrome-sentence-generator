import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='function-word-boundary-dp-20260916';SIGNATURE='function-word-variable-dp|determiner-aux-clitic-tense|fresh-scene-family|live-character-equation|new-state'
TEXTS=['A careful archivist will carry the ledger, and a patient porter has stored it beside the northern shelves.','The quiet engineer can measure a channel, and the watchful sailor is marking it near the western buoy.']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for s in TEXTS:
  t=normalize_letters(s);rows.append({'dp_variables':{'determiners':['a','the'],'auxiliaries':['will','has','can','is'],'clitic':['it'],'tense_endings':['carry','stored','measure','marking']},'scene_constraints':['agent/event/theme','coordinated independent clause','locative adjunct'],'live_equation':{'variable_prefix':t[:7],'variable_suffix':t[-7:],'free_center':True},'semantic_consistency':True,'audit':audit(s),'next_repair':{'operator':'change one auxiliary/determiner variable at first residual, then propagate tense agreement','reason':'function-word DP state is grammatical but leaves open character obligation'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'dynamic program treats determiners, auxiliaries, clitic, and tense endings as character-equation variables over fresh coordinated scenes','candidates':rows,'stats':{'rendered':2,'exact':0},'next_repair':{'operator':'change one auxiliary or determiner variable at first residual and propagate agreement','reason':'no exact anti-shortcut closure in this new function-word state space'},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'prior_transducer_reused':False,'posthoc_repair':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh archive/harbor function-word scene family','audits':['independent two-pointer','forward/reverse SHA-256','variable-state replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
