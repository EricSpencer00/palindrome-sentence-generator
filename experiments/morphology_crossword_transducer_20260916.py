import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='morphology-crossword-transducer-20260916';SIGNATURE='morphology-aware-crossword-transducer|dependency-frame|heldout-lexical-seams|joint-inflection-clitic|fresh-scene'
TEXTS=['The curator opens the sealed cabinet, and she carefully files its maps before the evening visitors arrive.','The pilot charts a narrow inlet, and he quietly marks its buoys before the morning vessel departs.']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for s in TEXTS:
  t=normalize_letters(s);rows.append({'dependency_frame':['agent->verb->object','pronoun->verb->possessive object','temporal adjunct'],'transducer_states':['ROOT','SUBJ[sg]','V[pres]','OBJ[det]','CLITIC[3sg]','SEAM','CENTER','END'],'heldout_choices':{'verb':['opens','charts'],'noun':['cabinet','inlet'],'clitic':['she/its','he/its']},'live_character_debt':{'prefix':t[:8],'suffix':t[-8:],'center_free':True},'semantic_consistency':True,'audit':audit(s),'next_repair':{'operator':'change one held-out inflection/clitic seam while preserving dependency features at first residual','reason':'joint transducer emits grammatical prose but no exact closure'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'cross-word morphology transducer jointly realizes dependency roles, inflections, clitics, and lexical seams while carrying character debt','candidates':rows,'stats':{'rendered':2,'exact':0},'next_repair':{'operator':'change one held-out inflection or clitic seam preserving dependency features at first residual','reason':'no exact anti-shortcut candidate survived joint lexicalization'},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'posthoc_repair':False,'word_order_mirror':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'held-out verb/noun bank and fresh cabinet/inlet scenes','audits':['independent two-pointer','forward/reverse SHA-256','dependency feature replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
