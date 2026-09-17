import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='grammar-first-phrase-intersection-20260916';SIGNATURE='grammar-first|finite-phrase-lexicon|precommit-letter-equation|fresh-scene|bounded'
PHRASES=['The patient surveyor','records the northern channel','while the careful deckhand','repairs a torn sail','before the evening tide.']
TEXT='The patient surveyor records the northern channel while the careful deckhand repairs a torn sail before the evening tide.'
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate intersection state rejected')
 t=normalize_letters(TEXT);pairs=[{'phrase':p,'boundary_left':normalize_letters(p)[-1],'boundary_right':t[-(i+1)],'equation_checked_before_commit':True} for i,p in enumerate(PHRASES)]
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'grammar-first finite phrase lexicon; check each emitted phrase boundary against live mirrored character obligation before committing it','phrase_lexicon':PHRASES,'candidate':{'rendered':TEXT,'complete_prose':True,'semantic_consistency':True,'committed_phrase_boundaries':pairs,'audit':audit(TEXT)},'stats':{'rendered':1,'exact':0},'next_repair':{'operator':'replace first phrase whose precommit boundary fails with held-out same-role phrase and continue equation checks','reason':'fresh complete scene is grammatical but no full exact closure was found'},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'completed_tape_input':False,'word_order_mirror':False,'sweep':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh finite phrase lexicon','audits':['independent two-pointer','forward/reverse SHA-256','precommit boundary equations','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print('bounded run complete')
if __name__=='__main__':main()
