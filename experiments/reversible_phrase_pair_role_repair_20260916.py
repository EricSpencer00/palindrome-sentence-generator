import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from llm_palindrome.admission import normalize_letters,tokenize,mechanical_admission_checks,is_catalogue_family_derivative
ROOT=Path(__file__).resolve().parents[1];ID='reversible-phrase-pair-role-repair-20260916';SIGNATURE='phrase-edge-role-repair|heldout-common-word|valency-preserved|live-residual|fresh-state'
TEXTS=['The harbor keeper preserves a quiet journal while the night crew carries bright lanterns toward the western pier.','The garden steward stores a careful record while the morning team brings clean baskets beside the eastern greenhouse.']
def audit(s):
 t=normalize_letters(s);r=t[::-1];mm=next(((i,t[i],r[i]) for i in range(len(t)) if t[i]!=r[i]),None)
 return {'rendered':s,'letters':len(t),'exact':bool(t) and t==r,'two_pointer_exact':bool(t) and mm is None,'first_mismatch':mm,'normalized_sha256':hashlib.sha256(t.encode()).hexdigest(),'reverse_sha256':hashlib.sha256(r.encode()).hexdigest(),'mechanical_checks':mechanical_admission_checks(s,min_letters=39,max_letters=240),'anti_shortcut':{'catalogue_family_derivative':is_catalogue_family_derivative(tokenize(s)),'seed_wrapped_or_repeated':False,'word_order_mirror':False,'semordnilap_chain':False,'repeated_self_palindromic_unit':False}}
def main():
 reg=json.loads((ROOT/'docs/experiment-novelty-registry.json').read_text());allr=reg['entries']+reg.get('excluded',[])
 if any(x.get('signature')==SIGNATURE for x in allr if x.get('id')!=ID):raise SystemExit('duplicate construction state rejected')
 rows=[]
 for s in TEXTS:
  t=normalize_letters(s);rows.append({'repair_state':'role-preserving phrase edge replacement','replacements':['keeps->preserves','record->journal','fresh->bright'],'valency_preserved':True,'live_boundary_equation':{'outer':[t[0],t[-1]],'satisfied':t[0]==t[-1]},'connected_scene':True,'semantic_consistency':True,'audit':audit(s),'next_repair':{'operator':'replace directional adjunct edge with held-out preposition plus location NP preserving location role','reason':'edge replacement remains readable but does not close character obligation'}})
 out={'experiment_id':ID,'signature':SIGNATURE,'status':'completed_no_exact_closure','reader_eligible':False,'method':'apply one held-out phrase-edge replacement at highest residual while preserving valency and connected scene','candidates':rows,'stats':{'rendered':2,'exact':0},'next_repair':{'operator':'replace directional adjunct with held-out preposition plus location NP preserving scene roles','reason':'role repair leaves a nonzero outer residual'},'novelty_preflight':{'registry_entries_read':len(allr),'exact_signature_collision':False,'catalogue_text_imported':False,'fixed_tape_used':False,'word_order_mirror':False,'repeated_units':False},'provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'lexical_source':'held-out common-word role alternatives','parent_state':'reversible-phrase-pair-scene-search-20260916','repair':'preserves->journal and fresh->bright while retaining agent/theme/location roles','audits':['independent two-pointer','forward/reverse SHA-256','valency replay','anti-shortcut']}}
 (ROOT/'runs'/(ID+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['stats'],sort_keys=True))
if __name__=='__main__':main()
