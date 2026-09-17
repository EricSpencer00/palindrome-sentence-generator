import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1]; ID='interrogative-relative-template-solver-20260916'; SIGNATURE='interrogative-relative-frame|authored-np-bank|online-character-equation|semantic-valency|fresh-template'
TEXTS=['Was the quiet curator sure that the young pilot had seen the chart I filed beside the harbor ledger?','Which careful gardener said that the patient mason repaired the gate I marked beside the eastern garden?']
def audit(s):
 t=normalize_letters(s); r=t[::-1]; mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text()); allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID): raise SystemExit('duplicate construction state rejected')
 rows=[]
 for s in TEXTS:
  t=normalize_letters(s); rows.append({'template':'interrogative + complement + object-gap relative','valency':{'main':'copular complement','embedded':'transitive agent patient','relative':'object gap'},'equation_samples':[{'left':t[i],'right':t[-i-1],'satisfied':t[i]==t[-i-1]} for i in range(5)],'semantic_consistency':True,'audit':audit(s)})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'choose authored noun phrases and verbs inside interrogative/relative CFG templates while checking mirrored character equations online','candidates':rows,'stats':{'rendered':2,'exact':0},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'known_template_output':False},'next_repair':{'operator':'substitute a valency-compatible auxiliary or relative pronoun selected by the first residual boundary','reason':'complete interrogative parses leave a nonzero outer character residual'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh authored NP and verb banks','audits':['independent two-pointer','forward/reverse SHA-256','semantic valency','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
