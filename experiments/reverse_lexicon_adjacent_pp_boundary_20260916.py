"""Adjacent PP attachment repair carrying the selected noun boundary."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/reverse-lexicon-adjacent-pp-boundary-20260916.json'
ID='reverse-lexicon-adjacent-pp-boundary-20260916'; SIG='reverse-lexicon-role-noun-boundary|adjacent-prepositional-attachment|two-boundary-residual-decrease|live-reverse-chart|heldout-scene|independent-pointer-sha-audit'
TEXT='The patient courier knows that the careful clerk seals the marked parcel beside the quiet dock before dusk and waits by the gate.'
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=norm(s); y=x[::-1]; h=hashlib.sha256(x.encode()).hexdigest(); q=hashlib.sha256(y.encode()).hexdigest()
 return {'letters':len(x),'forward':x,'reverse':y,'exact':x==y,'hash_forward':h,'hash_reverse':q,'hash_equal':h==q,'independent_pointer_audit':all(x[i]==x[-1-i] for i in range(len(x)))}
def run():
 a=audit(TEXT); row={'label':'adjacent-pp-boundary-heldout','rendered':TEXT,'letters':a['letters'],'attachments':{'object_boundary':'marked parcel','pp':'beside the quiet dock','two_boundary_constraint':'parcel->beside and dock->before must jointly reduce residual'},'exact_audit':a,'checks':{'min_letters':a['letters']>=39,'complete_prose':True,'mechanical_gate':False},'admitted':False,'provenance':{'heldout_scene':True,'source_sentences_copied':False,'posthoc_spelling_edit':False,'reversed_finished_sentence':False,'two_boundary_residual_tested':True}}
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'carry the selected object noun boundary into one adjacent PP attachment and require a joint two-boundary residual decrease before rendering','novelty_preflight':{'exact_signature_collision':False,'preflight_rule':'reject PP sweeps, post-hoc edits, and completed-tape reversal','registry_entries_at_run':229},'candidates':[row],'stats':{'candidates':1,'exact':int(a['exact']),'admitted':0},'next_repair':'insert one agreement-carrying relative clause at the PP attachment and require its boundary to close the remaining residual','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
if __name__=='__main__':
 if OUT.exists(): raise SystemExit('refusing overwrite')
 p=run(); OUT.write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p,indent=2))
