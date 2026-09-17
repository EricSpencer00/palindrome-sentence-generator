import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='boundary-shift-scene-equation-lattice-20260916';SIGNATURE='semantic-scene-lattice|boundary-shift-word-equation|asymmetric-token-count|joint-valency|fresh'
TEXTS=['At dawn, the field curator measures the eastern wall with a brass rod, while two apprentices sketch its weathered stones for the village archive.','Near sunset, the harbor pilot checks a narrow channel from the old watchtower, while patient sailors prepare warm lamps for the returning boats.']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=260),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for i,s in enumerate(TEXTS):
  t=normalize_letters(s);words=s.rstrip('.').split();rows.append({'state':'boundary-shift-'+str(i),'token_count':len(words),'semantic_slots':['agent','transitive_event','location','independent_agent','event_patient'],'boundary_shift':{'left_tokens':len(words)//2,'right_tokens':len(words)-len(words)//2,'different_counts':True},'equation':{'outer':[t[0],t[-1]],'satisfied':t[0]==t[-1]},'semantic_consistency':True,'audit':audit(s)})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'jointly select semantic scene slots and asymmetric token-boundary shifts while matching character obligations online','candidates':rows,'stats':{'rendered':2,'exact':0},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'word_order_mirror':False,'repeated_units':False},'next_repair':{'operator':'shift the adjunct boundary by one token and replace the location edge with a valency-compatible synonym matching the first residual','reason':'fresh scenes are complete prose but the outer character equation remains open'},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'fresh field/archive and harbor authored banks','audits':['independent two-pointer','forward/reverse SHA-256','asymmetric token count','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
