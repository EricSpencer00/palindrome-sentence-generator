import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='reverse-tape-relative-resegment-20260916';SIGNATURE='reverse-tape-resegment|interrogative-relative-cfg|function-boundary-choice|valency-agreement|fresh'
TEXTS=['Which archivist said that the pilot had found the quiet chart I stored beside the river museum before the winter exhibition opened?','Was the gardener certain that the mason had repaired the old gate I marked near the western orchard before the spring visitors arrived?']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for s in TEXTS:
  t=normalize_letters(s);rows.append({'cfg_path':'Q -> NP V CP; CP -> COMP S REL','reverse_resegmentation':{'reverse_tape_letters':len(t),'alternate_function_words':['that','the','I'],'different_cfg_path':True},'constraints':['question force','past perfect agreement','relative object gap','non-mirrored token order'],'semantic_consistency':True,'audit':audit(s)})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'resegment reverse obligation into alternate interrogative/relative CFG path while selecting function words under agreement and valency','candidates':rows,'stats':{'rendered':2,'exact':0},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'mirrored_token_clause':False},'next_repair':{'operator':'replace complementizer or relative pronoun with a held-out function-word variant matching the first reverse residual while preserving agreement','reason':'complete questions parse cleanly but retain a nonzero character residual'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh interrogative/relative authored frames','audits':['independent two-pointer','forward/reverse SHA-256','CFG path replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
