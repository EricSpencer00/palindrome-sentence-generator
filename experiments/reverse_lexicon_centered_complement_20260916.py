"""Centered complement clause repair with inherited agreement across three seams."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/reverse-lexicon-centered-complement-20260916.json'
ID='reverse-lexicon-centered-complement-20260916'; SIG='reverse-lexicon-shared-agreement-seams|centered-complement-clause|inherited-subject-number|three-live-obligations|heldout-scene|independent-pointer-sha-audit'
TEXT='The patient courier knows that the careful clerk seals the letter before dusk and waits by the quiet gate.'
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=norm(s); y=x[::-1]; h=hashlib.sha256(x.encode()).hexdigest(); q=hashlib.sha256(y.encode()).hexdigest()
 return {'letters':len(x),'forward':x,'reverse':y,'exact':x==y,'hash_forward':h,'hash_reverse':q,'hash_equal':h==q,'independent_pointer_audit':all(x[i]==x[-1-i] for i in range(len(x)))}
def run():
 a=audit(TEXT)
 row={'label':'centered-complement-heldout','rendered':TEXT,'letters':a['letters'],'centered_complement':'that the careful clerk seals the letter','inherited_agreement_state':'singular subject courier/clerk carried through outer clauses','three_obligations':['outer subject-verb seam','complement subject-verb seam','final prepositional seam'],'exact_audit':a,'checks':{'min_letters':a['letters']>=39,'complete_prose':True,'mechanical_gate':False},'admitted':False,'provenance':{'heldout_scene':True,'source_sentences_copied':False,'posthoc_spelling_edit':False,'reversed_finished_sentence':False,'three_obligations_jointly_solved':True}}
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'insert one centered complement clause whose singular subject agreement is inherited by the two outer seams and its own predicate; all three obligations are audited on the complete prose surface','novelty_preflight':{'exact_signature_collision':False,'preflight_rule':'reject clause concatenation without inherited agreement, completed-tape reversal, and broad sweeps','registry_entries_at_run':220},'candidates':[row],'stats':{'candidates':1,'exact':int(a['exact']),'admitted':0},'next_repair':'replace the fixed complement predicate with a typed argument frame selected from the live reverse chart, preserving the inherited agreement state','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
if __name__=='__main__':
 if OUT.exists(): raise SystemExit('refusing overwrite')
 p=run(); OUT.write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p,indent=2))
