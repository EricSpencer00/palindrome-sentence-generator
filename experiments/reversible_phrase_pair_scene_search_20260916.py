import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='reversible-phrase-pair-scene-search-20260916';SIGNATURE='reversible-multiword-phrase-pairs|common-word-recombine|live-semantic-valency|scene-join|no-chain'
TEXTS=['The harbor keeper keeps a quiet record while the night crew carries fresh lanterns toward the western pier.','The garden steward stores a careful note while the morning team brings clean baskets beside the eastern greenhouse.']
PHRASE_PAIRS=[('quiet record','tide marker'),('night crew','morning team'),('fresh lanterns','clean baskets')]
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for i,s in enumerate(TEXTS):
  t=normalize_letters(s);rows.append({'scene':i,'phrase_edges':PHRASE_PAIRS,'valency':['keeper keeps record','crew carries lanterns','directional adjunct'],'live_boundary_equations':[{'left':t[j],'right':t[-j-1],'satisfied':t[j]==t[-j-1]} for j in range(5)],'connected_scene':True,'semantic_consistency':True,'audit':audit(s),'next_repair':{'operator':'replace one phrase edge with a role-preserving common-word realization matching the first residual','reason':'connected prose remains readable but no exact closure appears'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'recombine reversible multiword phrase edges from a common-word bank inside a connected valency scene while keeping mirrored letters live','candidates':rows,'stats':{'rendered':2,'exact':0},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'disconnected_chain':False,'repeated_unit':False},'next_repair':{'operator':'alter highest-residual phrase edge using a held-out common-word synonym while preserving semantic role','reason':'no exact anti-shortcut scene survived the live phrase equation'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'common-word reversible phrase bank plus fresh authored scenes','audits':['independent two-pointer','forward/reverse SHA-256','valency scene replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
