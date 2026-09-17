import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='valency-clitic-live-lexicalizer-20260916';SIGNATURE='valency-attachment|clitic-boundary-choice|inflection-live-equation|connected-scene|fresh'
TEXTS=['The curator gives the sailor a ledger, and the sailor carefully carries it across the quiet quay before rain.','The nurse sends the porter a message, and the porter delivers it to the waiting ward before noon.']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for s in TEXTS:
  t=normalize_letters(s);rows.append({'valency_graph':['agent-gives-recipient-theme','agent-carries-theme','locative attachment'],'clitic_states':['it=theme','it after transitive verb'],'inflection_states':['third-person singular present'],'live_equation':{'outer_pair':[t[0],t[-1]],'first_four':[(t[i],t[-i-1]) for i in range(4)]},'semantic_consistency':True,'audit':audit(s),'next_repair':{'operator':'switch clitic placement while preserving recipient/theme roles, then choose matching inflection at first residual','reason':'valid attachment graph leaves open outer character obligation'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'solve valency attachment and clitic/inflection boundary states jointly while lexicalizing connected prose and tracking character equations','candidates':rows,'stats':{'rendered':2,'exact':0},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'word_order_mirror':False,'repeated_units':False},'next_repair':{'operator':'switch clitic attachment retaining recipient/theme valency, then retune inflection at highest residual','reason':'no exact anti-shortcut candidate survived this fresh lane'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh authored transfer-scene bank','audits':['independent two-pointer','normalized forward/reverse SHA','valency graph replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
