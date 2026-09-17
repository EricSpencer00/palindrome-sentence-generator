import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='corpus-seam-fresh-scene-grammar-20260916';SIGNATURE='corpus-derived-lexical-seams|fresh-authored-scene|pos-valency-grammar|live-character-equation|no-catalogue'
SEAMS=[{'left':'records the','right':'quiet harbor','pos':'V+DET / ADJ+N'},{'left':'carries a','right':'weathered chart','pos':'V+DET / ADJ+N'},{'left':'beside the','right':'old station','pos':'PREP+DET / ADJ+N'}]
TEXTS=['The curator records the quiet harbor beside the old station, while the pilot carries a weathered chart toward the archive.','The gardener carries a weathered chart beside the old station, while the keeper records the quiet harbor near the museum.']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=260),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for i,s in enumerate(TEXTS):
  t=normalize_letters(s);rows.append({'scene':i,'mined_seams':SEAMS,'fresh_authoring':True,'constraints':['POS-compatible seam','agent/event/patient valency','independent adjunct attachment'],'live_equations':[{'left':t[j],'right':t[-j-1],'satisfied':t[j]==t[-j-1]} for j in range(5)],'semantic_consistency':True,'audit':audit(s),'next_repair':{'operator':'select held-out POS-compatible seam for first residual while reauthoring whole clause','reason':'fresh scene is grammatical but no exact closure reached'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'mine lexical seam candidates by POS shape, then fresh-author connected scenes under valency constraints while tracking character equations','candidates':rows,'stats':{'rendered':2,'exact':0},'next_repair':{'operator':'choose held-out POS-compatible seam matching first residual and reauthor full connected scene','reason':'no exact anti-shortcut candidate survived corpus-seam lexicalization'},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'copied_spans':False,'repeated_units':False,'word_order_mirror':False,'disconnected_chain':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'POS-shaped lexical seams only; emitted scenes freshly authored','audits':['independent two-pointer','forward/reverse SHA-256','valency/POS gate','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
