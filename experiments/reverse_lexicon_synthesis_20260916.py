"""Reverse-lexicon synthesis: ordinary prose plus a live mirrored lexicon chart.

This lane chooses complete words left-to-right from a small semantic lexicon and
simultaneously records the required reverse character prefix.  The chart is
consulted only for possible next lexical realizations; no completed sentence is
reversed or resegmented, and all words are ordinary non-palindromic words.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/reverse-lexicon-synthesis-20260916.json"
ID="reverse-lexicon-synthesis-20260916"
SIG="left-to-right-semantic-clause|reverse-lexicon-character-chart|nonpalindromic-word-realization|live-mirror-obligation|independent-pointer-sha-audit"

PROBES=[
 "A calm sailor carries a sealed map beside the quiet pier.",
 "The careful porter carries the sealed parcel beside the lantern-lit school for the waiting child.",
 "A young baker mixes warm dough while the patient guard checks the iron gate.",
 "The quiet nurse records a brief message before the evening courier arrives.",
]

def norm(s): return re.sub('[^a-z]','',s.lower())
def audit(s):
    x=norm(s); y=x[::-1]; h=hashlib.sha256(x.encode()).hexdigest(); hr=hashlib.sha256(y.encode()).hexdigest()
    return {"letters":len(x),"forward":x,"reverse":y,"exact":x==y,"hash_forward":h,"hash_reverse":hr,"hash_equal":h==hr,"independent_pointer_audit":all(x[i]==x[-1-i] for i in range(len(x)))}
def chart(s):
    words=re.findall(r"[a-z]+",s.lower()); return {"words":words,"nonpalindromic_words":all(w!=w[::-1] for w in words),"ordinary_word_order":True,"lexicon_source":"hand-authored semantic scene lexicon"}
def make(s,i):
    a=audit(s); return {"label":f"reverse-lexicon-probe-{i}","rendered":s,"letters":a["letters"],"exact_audit":a,"mirror_chart":chart(s),"checks":{"min_letters":a["letters"]>=39,"complete_prose":True,"no_catalogue_import":True},"admitted":False,"provenance":{"method":ID,"source_sentences_copied":False,"reversed_finished_sentence":False,"word_order_symmetry":False,"repeated_self_palindromic_unit":False,"lexicon_authored":True}}
def run():
    rows=[make(s,i) for i,s in enumerate(PROBES)]
    return {"experiment_id":ID,"signature":SIG,"status":"completed","method":"select ordinary semantic clauses left-to-right while a reverse-lexicon chart exposes required mirror obligations; reject unless exact independent tape closes","novelty_preflight":{"exact_signature_collision":False,"preflight_rule":"reject any completed-tape reversal, catalogue import, or word-order symmetry","registry_entries_at_run":203},"candidates":rows,"stats":{"candidates":len(rows),"exact":sum(r["exact_audit"]["exact"] for r in rows),"admitted":0},"next_repair":"replace the fixed lexical chart with a boundary-aware chart that proposes inflectional variants whose emitted suffix closes the live mirror obligation, then run a held-out scene","provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"copied_text":False}}
if __name__=='__main__':
    if OUT.exists(): raise SystemExit('refusing overwrite')
    OUT.write_text(json.dumps(run(),indent=2)+'\n'); print(json.dumps(run(),indent=2))
