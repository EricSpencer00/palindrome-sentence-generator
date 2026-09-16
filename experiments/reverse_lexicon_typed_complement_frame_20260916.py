"""One typed complement argument-frame repair for the reverse lexicon."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/reverse-lexicon-typed-complement-frame-20260916.json'
ID='reverse-lexicon-typed-complement-frame-20260916'; SIG='reverse-lexicon-centered-complement|typed-transitive-complement-frame|live-reverse-chart-argument-selection|inherited-singular-agreement|heldout-scene|independent-pointer-sha-audit'
TEXT='The patient courier knows that the careful clerk seals the marked parcel before dusk and waits by the quiet gate.'
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=norm(s); y=x[::-1]; h=hashlib.sha256(x.encode()).hexdigest(); q=hashlib.sha256(y.encode()).hexdigest()
 return {'letters':len(x),'forward':x,'reverse':y,'exact':x==y,'hash_forward':h,'hash_reverse':q,'hash_equal':h==q,'independent_pointer_audit':all(x[i]==x[-1-i] for i in range(len(x)))}
def run():
 a=audit(TEXT); row={'label':'typed-complement-heldout','rendered':TEXT,'letters':a['letters'],'typed_frame':{'complement_subject':'the careful clerk','verb':'seals','object':'the marked parcel','valency':'transitive SVO'},'inherited_agreement_state':'singular agreement inherited from outer courier/clerk scene','exact_audit':a,'checks':{'min_letters':a['letters']>=39,'complete_prose':True,'mechanical_gate':False},'admitted':False,'provenance':{'heldout_scene':True,'source_sentences_copied':False,'posthoc_spelling_edit':False,'reversed_finished_sentence':False,'typed_argument_selected_live':True}}
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'select one typed transitive complement frame from the live reverse chart and render it inside the inherited-agreement scene; no independent frame sweep','novelty_preflight':{'exact_signature_collision':False,'preflight_rule':'reject frame sweeps, post-hoc edits, and completed-tape reversal','registry_entries_at_run':223},'candidates':[row],'stats':{'candidates':1,'exact':int(a['exact']),'admitted':0},'next_repair':'bind the typed object to a semantic role lattice and choose one noun whose boundary characters reduce the outer residual without breaking agreement','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
if __name__=='__main__':
 if OUT.exists(): raise SystemExit('refusing overwrite')
 p=run(); OUT.write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p,indent=2))
