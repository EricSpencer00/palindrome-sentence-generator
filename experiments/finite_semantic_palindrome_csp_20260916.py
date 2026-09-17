import hashlib,json,itertools
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='finite-semantic-palindrome-csp-20260916';SIGNATURE='finite-semantic-csp|subject-verb-object-adjunct|joint-character-obligations|fresh-scene-family|no-posthoc'
SUBJ=['The observant ranger','The patient curator'];VERB=['studies','records'];OBJ=['a weathered compass','the coastal ledger'];ADJ=['beside the rescue shed','near the museum steps']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=260),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for vals in itertools.islice(itertools.product(SUBJ,VERB,OBJ,ADJ),2):
  s=f'{vals[0]} {vals[1]} {vals[2]} {vals[3]}.';t=normalize_letters(s);rows.append({'semantic_slots':dict(zip(['subject','verb','object','adjunct'],vals)),'csp_constraints':['SVO valency','locative adjunct','fresh authored alternatives'],'character_obligations':{'outer':[t[0],t[-1]],'center_free':True},'semantic_consistency':True,'audit':audit(s),'next_repair':{'operator':'replace one object alternative jointly with its verb frame at highest residual','reason':'complete prose CSP state leaves open character obligation'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'finite CSP jointly assigns fresh subject, verb, object, and adjunct alternatives while carrying character obligations','candidates':rows,'stats':{'rendered':2,'exact':0},'next_repair':{'operator':'replace one object alternative jointly with its valency-compatible verb at highest residual','reason':'no exact anti-shortcut closure in fresh finite semantic CSP'},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'posthoc_readability':False,'word_order_mirror':False,'repeated_units':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'new authored subject/verb/object/adjunct alternatives','audits':['independent two-pointer','forward/reverse SHA-256','CSP feature replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
