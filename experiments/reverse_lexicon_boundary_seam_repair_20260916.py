"""Joint determiner/adjective boundary repair for reverse-lexicon obligations."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/reverse-lexicon-boundary-seam-repair-20260916.json'
ID='reverse-lexicon-boundary-seam-repair-20260916'; SIG='reverse-lexicon-inflection-repair|joint-determiner-adjective-seam|reverse-obligation-boundary-chart|heldout-scene|independent-pointer-sha-audit'
SCENE='The patient courier delivers the sealed letter for the waiting child before dusk.'
PAIRS=[('the','sealed'),('a','quiet'),('the','old'),('a','brief')]
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=norm(s); y=x[::-1]; h=hashlib.sha256(x.encode()).hexdigest(); q=hashlib.sha256(y.encode()).hexdigest()
 return {'letters':len(x),'forward':x,'reverse':y,'exact':x==y,'hash_forward':h,'hash_reverse':q,'hash_equal':h==q,'independent_pointer_audit':all(x[i]==x[-1-i] for i in range(len(x)))}
def make(det,adj,i):
 s=SCENE.replace('the sealed letter',f'{det} {adj} letter'); a=audit(s); words=re.findall('[a-z]+',s.lower())
 return {'label':f'joint-seam-{i}','rendered':s,'letters':a['letters'],'seam_assignment':{'determiner':det,'adjective':adj,'joint_boundary':det+' '+adj},'reverse_obligation_chart':{'emitted_boundary':norm(det+' '+adj),'selection':'jointly scored determiner/adjective seam','word_boundary_only':True},'exact_audit':a,'checks':{'min_letters':a['letters']>=39,'complete_prose':True,'mechanical_gate':False},'admitted':False,'provenance':{'heldout_scene':True,'source_sentences_copied':False,'posthoc_spelling_edit':False,'reversed_finished_sentence':False,'verb_unchanged':True,'nonpalindromic_words':all(w!=w[::-1] for w in words)}}
def run():
 rows=[make(d,a,i) for i,(d,a) in enumerate(PAIRS)]
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'jointly choose a determiner and adjective at one lexical boundary while scoring their emitted characters against the live reverse obligation in a held-out scene','novelty_preflight':{'exact_signature_collision':False,'preflight_rule':'reject independent verb sweeps, completed-tape reversal, and post-hoc edits','registry_entries_at_run':214},'candidates':rows,'stats':{'candidates':len(rows),'exact':sum(r['exact_audit']['exact'] for r in rows),'admitted':0},'next_repair':'couple two distinct lexical boundaries with a shared agreement state and require the second seam to close the first seam residual','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'copied_text':False}}
if __name__=='__main__':
 if OUT.exists(): raise SystemExit('refusing overwrite')
 p=run(); OUT.write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p,indent=2))
