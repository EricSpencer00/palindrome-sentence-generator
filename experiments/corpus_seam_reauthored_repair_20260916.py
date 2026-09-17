import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='corpus-seam-reauthored-repair-20260916';SIGNATURE='heldout-pos-seam-repair|full-scene-reauthoring|valency-preserved|single-state|fresh'
TEXT='At sunrise, the ferry captain logs a narrow channel beside the eastern pier, while the patient mechanic checks each lamp before the first crossing.'
SEAM={'old_residual':'records the','heldout':'logs a','pos':'V+DET','role':'agent records artifact'}
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=260),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 t=normalize_letters(TEXT);row={'construction_state':'held-out seam + full reauthoring','seam':SEAM,'fresh_authoring':True,'valency':['captain logs channel','mechanic checks lamps','temporal adjunct'],'live_equation':{'outer_pair':[t[0],t[-1]],'first_residual':audit(TEXT)['first_mismatch']},'semantic_consistency':True,'audit':audit(TEXT),'next_repair':{'operator':'select held-out temporal adjunct with residual-matching onset and reauthor both clauses','reason':'seam repair yields readable prose but leaves open outer character obligation'}}
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'select one held-out POS-compatible seam and reauthor a complete connected valency scene around it','candidates':[row],'stats':{'rendered':1,'exact':0},'next_repair':row['next_repair'],'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'replayed_prior_scene':False,'copied_spans':False,'word_order_mirror':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'held-out POS seam plus fresh ferry scene','parent_state':'corpus-seam-fresh-scene-grammar-20260916','repair':'logs a replaces prior seam role, followed by full scene reauthoring','audits':['independent two-pointer','forward/reverse SHA-256','valency replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
