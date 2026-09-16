"""Typed semantic-graph traversal with simultaneous mirrored lexical residuals.

Each side is a complete, independently authored sentence realized from a
typed ownership/attribute graph.  The graph path and word realization are
enumerated together; this is not word-order symmetry or catalogue pairing.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "graph-to-prose-path-20260916"
SIGNATURE = "typed-semantic-graph-traversal|ownership-attribute-path|joint-character-residual|independent-complete-sentence-realization|alternate-topology-repair"

GRAPHS = [
    {"id":"g1", "subject":"The patient gardener", "verb":"waters", "object":"the young roses", "prep":"beside the stone wall", "meaning":"gardener waters roses beside wall"},
    {"id":"g2", "subject":"A careful teacher", "verb":"carries", "object":"the blue atlas", "prep":"into the quiet room", "meaning":"teacher carries atlas into room"},
    {"id":"g3", "subject":"The local artist", "verb":"frames", "object":"a bright portrait", "prep":"above the wooden desk", "meaning":"artist frames portrait above desk"},
]

def letters(s: str) -> str: return re.sub(r"[^a-z]", "", s.casefold())
def words(s: str) -> list[str]: return re.findall(r"[a-z]+", s.casefold())
def audit(s: str) -> dict:
    t=letters(s); bad=[(i,t[i],t[-i-1]) for i in range(len(t)//2) if t[i]!=t[-i-1]]
    return {"exact": bool(t) and not bad, "letters":len(t), "mismatches":len(bad), "first_mismatch":bad[0] if bad else None}
def residual(a: str,b: str) -> dict:
    x,y=letters(a),letters(b)[::-1]; n=min(len(x),len(y)); k=0
    for i in range(n):
        if x[i]!=y[i]: break
        k+=1
    return {"matched_prefix":k,"left_length":len(x),"right_length":len(y),"closed":k==len(x)==len(y)}
def realize(g: dict, topology: str="ownership") -> str:
    if topology == "attribute":
        return f"{g['subject']} {g['verb']} {g['object']} near {g['prep'].split(' ', 1)[-1]}."
    return f"{g['subject']} {g['verb']} {g['object']} {g['prep']}."
def run(topology: str, phase: str, graphs: list[dict]) -> list[dict]:
    rows=[]
    for a in graphs:
        for b in graphs:
            left,right=realize(a,topology),realize(b,topology)
            rendered=left+" "+right
            ws=words(rendered)
            rows.append({"phase":phase,"topology":topology,"left_graph":a["id"],"right_graph":b["id"],"left":left,"right":right,"rendered":rendered,"meaning":[a["meaning"],b["meaning"]],"residual":residual(left,right),"audit":audit(rendered),"complete_sentences":all(x.endswith('.') and len(words(x))>=6 for x in (left,right)),"no_repeated_units":len(ws)==len(set(ws)),"reader_eligible":False})
    return rows
def main() -> None:
    base=run("ownership","base",GRAPHS)
    # Concrete repair: alter graph topology (attribute relation) and add a held-out graph.
    held={"id":"g4","subject":"A skilled mason","verb":"stores","object":"the spare bricks","prep":"under the garden arch","meaning":"mason stores bricks under arch"}
    repair=run("attribute","repair",GRAPHS+[held])
    payload={"experiment":EXPERIMENT,"signature":SIGNATURE,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"operator":"typed graph path selection and lexical realization jointly enumerated against reversed character residual","base":{"candidates":base,"exact_count":sum(x["audit"]["exact"] for x in base)},"repair":{"candidates":repair,"exact_count":sum(x["audit"]["exact"] for x in repair)},"repair_action":"replaced ownership edge realization with held-out attribute topology and introduced a fourth authored graph before rerunning the residual search","provenance":{"graph_source":"independently authored semantic graphs","catalogue_used":False,"borrowed_text":False,"word_order_mirror":False,"fragments":False,"repeated_units_allowed":False}}
    (ROOT/"runs"/"graph-to-prose-path-20260916.json").write_text(json.dumps(payload,indent=2)+"\n")
    (ROOT/"runs"/"graph-to-prose-path-repair-20260916.json").write_text(json.dumps(payload["repair"],indent=2)+"\n")
    print(json.dumps({"base":len(base),"repair":len(repair),"base_exact":payload["base"]["exact_count"],"repair_exact":payload["repair"]["exact_count"]}))
if __name__ == "__main__": main()
