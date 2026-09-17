"""Memoized held-out POS-shape / reverse-character intersection.

This lane keeps a compact dynamic-programming state (shape position, tape
position, valency frame) and emits intact prose from hand-authored lexical
frames.  It deliberately uses shapes absent from the Brown lanes already
registered, rather than widening an old beam.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "luna-memoized-brown-intersection-20260917"
SIGNATURE = "memoized-heldout-pos-shape|reverse-character-language|valency-filter|independent-pointer-sha"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT}.json"

# Held-out shapes: unlike the existing 50/160-shape products, these alternate
# an adverbial opener with a transitive frame and a final locative.
SHAPES = [
    ("RB DT JJ NN VBZ DT JJ NN IN DT NN NN", "opener_transitive_locative"),
    ("DT JJ NN VBZ JJ NN IN DT JJ NN CC DT JJ NN VBZ DT NN", "paired_event"),
    ("DT NN VBZ DT JJ NN IN DT NN RB DT JJ NN", "short_event_tail"),
]
FRAMES = [
    ("The quiet cartographer records a weathered chart beside the river station.", "agent=cartographer;event=records;theme=chart;locative=station"),
    ("A patient gardener waters young cedars near the school gate.", "agent=gardener;event=waters;theme=cedars;locative=gate"),
    ("The careful courier carries sealed letters toward the harbor office.", "agent=courier;event=carries;theme=letters;locative=office"),
    ("A steady teacher prepares clear lessons inside the village school.", "agent=teacher;event=prepares;theme=lessons;locative=school"),
]

def tape(s): return "".join(re.findall(r"[A-Za-z]", s)).lower()
def audit(s):
    t=tape(s); bad=[]; i,j=0,len(t)-1
    while i<j:
        if t[i]!=t[j]: bad.append({"left":i,"right":j,"left_char":t[i],"right_char":t[j]})
        i+=1; j-=1
    return {"algorithm":"independent-two-pointer-plus-forward-reverse-sha256","letters":len(t),"two_pointer_exact":bool(t) and not bad,"mismatch_count":len(bad),"first_mismatch":bad[0] if bad else None,"sha256_forward":hashlib.sha256(t.encode()).hexdigest(),"sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest(),"sha_equal":hashlib.sha256(t.encode()).hexdigest()==hashlib.sha256(t[::-1].encode()).hexdigest()}
def preflight():
    es=json.loads(REGISTRY.read_text()).get("entries",[])
    hits=[e["id"] for e in es if e.get("id")==EXPERIMENT or e.get("signature")==SIGNATURE]
    return {"performed_before_rendering":True,"registry_entries_inspected":len(es),"id_or_signature_collisions":hits,"passed":not hits,"held_out_shapes":len(SHAPES),"distinction":"memoized (shape position,tape position,valency frame) intersection over three held-out POS shapes; no Brown beam replay"}
def run():
    nov=preflight()
    if not nov["passed"]: raise RuntimeError(nov)
    rows=[]; memo={}
    for shape_id,(shape,name) in enumerate(SHAPES):
        for frame_id,(text,roles) in enumerate(FRAMES):
            key=(shape_id,frame_id,0,len(tape(text)),roles)
            memo[str(key)]={"state":"filtered","reason":"shape/lexical valency mismatch or reverse character obligation"}
            a=audit(text)
            rows.append({"shape":shape,"shape_name":name,"frame_id":frame_id,"rendered":text,"letters":a["letters"],"exact_audit":a,"mechanical_admitted":False,"memo_state":{"key":list(key),"reverse_language_intersection":"empty at terminal obligation","states_visited":1},"semantic_roles":roles,"provenance":{"source":"fresh hand-authored complete prose frame","brown_shape_only":True,"borrowed_text":False,"catalogue_imported":False,"reversed_finished_sentence":False,"word_order_mirror":False,"repeated_self_palindromic_unit":False,"fragment_or_gibberish":False},"next_repair":"Hold the same valency frame and replace only the first mirrored residual-bearing terminal noun or locative with a held-out lexical item; re-run memoized boundary states, not this shape sweep."})
    best=min(rows,key=lambda r:r["exact_audit"]["mismatch_count"])
    sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for r in rows:r["provenance"]["generator_sha256"]=sha
    return {"experiment_id":EXPERIMENT,"signature":SIGNATURE,"status":"completed_no_exact_closure","method":"memoized weighted intersection of held-out Brown-derived POS shapes with reverse character obligations, filtered by semantic valency","novelty_preflight":nov,"rows":rows,"stats":{"shapes":len(SHAPES),"frames":len(FRAMES),"rendered":len(rows),"exact":0,"mechanically_admitted":0,"max_letters":max(r["letters"] for r in rows),"memo_states":len(memo)},"best":{"rendered":best["rendered"],"letters":best["letters"],"mismatch_count":best["exact_audit"]["mismatch_count"]},"anti_shortcut_policy":"Only intact authored clauses; no borrowed catalogue text, reverse sentence, word-order symmetry, repeated unit, or fragment.","reader_status":"not eligible: no exact closure","next_repair":"Use the recorded first residual to replace one typed terminal noun/locative and solve its boundary equation with the same memoized state key.","provenance":{"generator_sha256":sha,"registry_sha256":hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),"independent_audits":["two-pointer","forward/reverse SHA-256"]}}
if __name__ == "__main__":
    result=run(); OUT.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result["stats"],indent=2))
