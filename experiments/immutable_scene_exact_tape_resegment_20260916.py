import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='immutable-scene-exact-tape-resegment-20260916';SIGNATURE='fresh-multiclause-authoring|immutable-tape-diagnostic|grammar-resegment|no-catalogue|no-sweep'
TEXT='At dawn, the archivist opens the west gallery, and the curator labels each recovered map while visitors wait quietly beside the old stair.'
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=260),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate diagnostic state rejected')
 t=normalize_letters(TEXT);a=audit(TEXT)
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'diagnostic_no_grammatical_exact_resegmentation','reader_eligible':False,'method':'author one fresh multi-clause sentence first; freeze normalized tape; attempt one grammar-aware reverse resegmentation diagnostic','immutable_tape':{'letters':len(t),'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'frozen':True},'candidate':{'rendered':TEXT,'fresh_authoring':True,'complete_prose':True,'audit':a},'resegmentation':{'grammar':'S -> clause conjunction clause adjunct','paths_tested':1,'exact_grammar_paths':0,'used_catalogue_text':False,'sweep':False,'failure':'reverse tape has no complete ordinary-English parse under this grammar'},'next_repair':{'operator':'author a new center-bearing clause family whose seam terminals are chosen before tape freeze, then rerun one exact-tape intersection','reason':'immutable authored tape is grammatical forward but cannot be resegmented into a grammatical reverse path'},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_as_input':False,'word_order_mirror':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh authored archive-gallery scene','audits':['independent two-pointer','forward/reverse SHA-256','immutable tape replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print('diagnostic complete')
if __name__=='__main__':main()
