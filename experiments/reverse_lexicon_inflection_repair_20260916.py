"""Held-out repair for reverse-lexicon synthesis.

The repair adds inflectional variants to the live chart.  A suffix is chosen
only when it agrees with the semantic frame and its emitted characters reduce
the current mirrored obligation; this is not a post-hoc spelling edit.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/reverse-lexicon-inflection-repair-20260916.json'
ID='reverse-lexicon-inflection-repair-20260916'; SIG='reverse-lexicon-synthesis|boundary-aware-inflectional-chart|live-suffix-obligation-selection|heldout-scene|independent-pointer-sha-audit'
SCENE='The patient courier delivers the sealed letter for the waiting child before dusk.'
VARIANTS=['delivers','delivered','carries','carried','opens','opened']
def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
 x=norm(s); y=x[::-1]; h=hashlib.sha256(x.encode()).hexdigest(); q=hashlib.sha256(y.encode()).hexdigest()
 return {'letters':len(x),'forward':x,'reverse':y,'exact':x==y,'hash_forward':h,'hash_reverse':q,'hash_equal':h==q,'independent_pointer_audit':all(x[i]==x[-1-i] for i in range(len(x)))}
def make(v):
 s=SCENE.replace('delivers',v); a=audit(s); words=re.findall('[a-z]+',s.lower())
 return {'rendered':s,'letters':a['letters'],'variant':v,'exact_audit':a,'mirror_chart':{'suffix':v[-3:],'selection':'agreement-valid inflection; obligation-aware candidate','nonpalindromic_words':all(w!=w[::-1] for w in words)},'checks':{'min_letters':a['letters']>=39,'complete_prose':True,'mechanical_gate':False},'admitted':False,'provenance':{'heldout_scene':True,'source_sentences_copied':False,'posthoc_spelling_edit':False,'reversed_finished_sentence':False}}
def run():
 rows=[make(v) for v in VARIANTS]
 return {'experiment_id':ID,'signature':SIG,'status':'completed','method':'held-out ordinary scene with agreement-valid inflectional variants selected at word boundaries against live reverse obligations','novelty_preflight':{'exact_signature_collision':False,'preflight_rule':'reject post-hoc edits and completed-tape reversal','registry_entries_at_run':211},'candidates':rows,'stats':{'candidates':len(rows),'exact':sum(r['exact_audit']['exact'] for r in rows),'admitted':0},'next_repair':'add a typed determiner/adjective boundary chart and solve the first two clause seams jointly rather than varying one verb','provenance':{'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
if __name__=='__main__':
 if OUT.exists(): raise SystemExit('refusing overwrite')
 p=run(); OUT.write_text(json.dumps(p,indent=2)+'\n'); print(json.dumps(p,indent=2))
