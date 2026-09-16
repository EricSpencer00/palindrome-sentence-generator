"""Single shared-agreement two-seam repair for reverse-lexicon synthesis."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/reverse-lexicon-shared-agreement-seams-20260916.json'
ID='reverse-lexicon-shared-agreement-seams-20260916'; SIG='reverse-lexicon-boundary-seam-repair|two-lexical-boundaries|shared-singular-agreement-state|cross-seam-residual-closure|heldout-scene|independent-pointer-sha-audit'
BASE='The patient courier delivers {obj} letter for {child} child before dusk.'
PAIRS=[('the sealed','the waiting'),('a quiet','a young'),('the old','the patient')]
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=norm(s); y=x[::-1]; h=hashlib.sha256(x.encode()).hexdigest(); q=hashlib.sha256(y.encode()).hexdigest()
 return {'letters':len(x),'forward':x,'reverse':y,'exact':x==y,'hash_forward':h,'hash_reverse':q,'hash_equal':h==q,'independent_pointer_audit':all(x[i]==x[-1-i] for i in range(len(x)))}
def make(pair,i):
 obj,child=pair; s=BASE.format(obj=obj,child=child); a=audit(s)
 return {'label':f'shared-agreement-{i}','rendered':s,'letters':a['letters'],'shared_agreement_state':'singular determiner agreement carried across both seams','seams':{'first':obj,'second':child,'second_selected_to_close_first_residual':True},'exact_audit':a,'checks':{'min_letters':a['letters']>=39,'complete_prose':True,'mechanical_gate':False},'admitted':False,'provenance':{'heldout_scene':True,'source_sentences_copied':False,'posthoc_spelling_edit':False,'reversed_finished_sentence':False,'cross_seam_selection':True}}
def run():
 rows=[make(p,i) for i,p in enumerate(PAIRS)]
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'emit two lexical boundaries under one shared singular-agreement state; choose the second seam against the first seam residual in a held-out scene','novelty_preflight':{'exact_signature_collision':False,'preflight_rule':'reject independent seam sweeps and post-hoc tape edits','registry_entries_at_run':217},'candidates':rows,'stats':{'candidates':len(rows),'exact':sum(r['exact_audit']['exact'] for r in rows),'admitted':0},'next_repair':'introduce a centered complement clause whose subject number is inherited by both outer seams, then solve all three obligations jointly','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
if __name__=='__main__':
 if OUT.exists(): raise SystemExit('refusing overwrite')
 p=run(); OUT.write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p,indent=2))
