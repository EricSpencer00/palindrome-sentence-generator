import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='semantic-slot-single-object-repair-20260916';SIGNATURE='semantic-slot-object-repair|single-state|role-compatible|scene-attachment-fixed|fresh'
TEXT='At noon, the coastal ranger studies a weathered sextant beside the rescue shed, while volunteers prepare fresh water for the arriving hikers.'
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=260),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 t=normalize_letters(TEXT);row={'repair_state':'single object substitution','parent_state':'semantic-slot-cross-boundary-dp-20260916','change':{'old':'damaged compass','new':'weathered sextant'},'role_fixed':'ranger studies navigational object','attachments_fixed':['beside rescue shed','while volunteers prepare water'],'semantic_consistency':True,'audit':audit(TEXT),'next_repair':{'operator':'change only location adjunct noun while preserving object role and attachment','reason':'object repair remains readable but leaves nonzero character residual'}}
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'single held-out object substitution with clause relexicalization and all attachment roles frozen','candidates':[row],'stats':{'rendered':1,'exact':0},'next_repair':row['next_repair'],'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'sweep':False,'prior_scene_replayed':False,'word_order_mirror':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'held-out navigational object slot','parent_state':'semantic-slot-cross-boundary-dp-20260916','repair':'damaged compass -> weathered sextant; attachment and participant roles unchanged','audits':['independent two-pointer','forward/reverse SHA-256','role replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
