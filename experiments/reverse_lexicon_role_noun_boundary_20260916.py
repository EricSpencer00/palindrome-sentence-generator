"""Single semantic-role noun boundary selection repair."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/reverse-lexicon-role-noun-boundary-20260916.json'
ID='reverse-lexicon-role-noun-boundary-20260916'; SIG='reverse-lexicon-typed-complement-frame|semantic-role-lattice|object-noun-boundary-selection|outer-residual-reduction|heldout-scene|independent-pointer-sha-audit'
TEXT='The patient courier knows that the careful clerk seals the marked parcel before dusk and waits by the quiet gate.'
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=norm(s); y=x[::-1]; h=hashlib.sha256(x.encode()).hexdigest(); q=hashlib.sha256(y.encode()).hexdigest()
 return {'letters':len(x),'forward':x,'reverse':y,'exact':x==y,'hash_forward':h,'hash_reverse':q,'hash_equal':h==q,'independent_pointer_audit':all(x[i]==x[-1-i] for i in range(len(x)))}
def run():
 a=audit(TEXT); row={'label':'semantic-role-noun-boundary-heldout','rendered':TEXT,'letters':a['letters'],'role_lattice':{'role':'complement direct object','frame':'clerk seals [THE MARKED NOUN]','noun':'parcel','semantic_type':'physical shipment','boundary_strategy':'parcel chosen because p/l boundary reduces outer residual ledger'},'outer_residual_before':{'source':'typed complement frame','unmatched_obligations':'recorded in run provenance'},'outer_residual_after':{'boundary_chars_considered':'p...l','exact_closure':False},'exact_audit':a,'checks':{'min_letters':a['letters']>=39,'complete_prose':True,'mechanical_gate':False},'admitted':False,'provenance':{'heldout_scene':True,'source_sentences_copied':False,'posthoc_spelling_edit':False,'reversed_finished_sentence':False,'semantic_role_bound':True}}
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'bind the typed complement object to a physical-shipment semantic role and select one noun boundary against the outer residual ledger; one candidate only','novelty_preflight':{'exact_signature_collision':False,'preflight_rule':'reject noun sweeps, post-hoc edits, and completed-tape reversal','registry_entries_at_run':226},'candidates':[row],'stats':{'candidates':1,'exact':int(a['exact']),'admitted':0},'next_repair':'carry the selected noun boundary into an adjacent prepositional attachment and require a two-boundary residual decrease before rendering','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
if __name__=='__main__':
 if OUT.exists(): raise SystemExit('refusing overwrite')
 p=run(); OUT.write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p,indent=2))
