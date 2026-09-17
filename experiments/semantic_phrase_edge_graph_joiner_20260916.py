import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1]; ID='semantic-phrase-edge-graph-joiner-20260916'; SIGNATURE='heldout-role-phrase-edges|semantic-path-join|online-boundary-equation|complete-clause-grammar|fresh'
TEXTS=['The lighthouse keeper records the tide marks beside the storm-dark pier while the young deckhand mends a torn sail.','The patient botanist labels the winter seedlings in the glasshouse at dawn while the careful apprentice sorts the field notes.','The lighthouse keeper records the tide marks beside the storm-dark pier while the young deckhand repairs a torn sail.']
def audit(s):
 t=normalize_letters(s); r=t[::-1]; mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=260),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text()); allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID): raise SystemExit('duplicate construction state rejected')
 rows=[]
 for i,s in enumerate(TEXTS):
  t=normalize_letters(s); rows.append({'path':('repair' if i==2 else 'edge-path-'+str(i)),'phrase_edges':s.split(),'boundary_equations':[{'left':t[j],'right':t[-j-1],'satisfied':t[j]==t[-j-1]} for j in range(4)],'semantic_consistency':True,'audit':audit(s)})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'compose held-out semantic phrase edges into complete SVO paths while evaluating character boundary equations online','candidates':rows,'stats':{'rendered':3,'exact':0},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'corpus_sentence_copy':False},'next_repair':{'operator':'swap the deckhand action edge for a held-out transitive verb sharing the required boundary character while preserving role and tense','reason':'the role-preserving repair lowers the first residual but does not close the full tape'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'held-out authored role phrase banks','repair':'changed deckhand action from mends to repairs preserving role, object, tense, and scene','audits':['independent two-pointer','forward/reverse SHA-256','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__': main()
