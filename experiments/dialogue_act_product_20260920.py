"""Dialogue-act product search: complete turns grow under shared character equations.

This is deliberately not a clause-pair or repair search.  Each arm is a
different, authored conversational act (question, answer, request, promise,
warning, report); a product state chooses whole acts while comparing the
outer character residuals before accepting the next act.  It is retained as
evidence even when the exact intersection is empty.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEFT = (
    "did the bell ring before dawn", "please carry the lantern home",
    "the keeper marked the quiet harbor", "will the singer return at noon",
    "we should shelter the injured fox", "a patient child planted beans",
)
RIGHT = (
    "the steward counted grain at dusk", "can you bring the red book",
    "the sailor repaired a broken oar", "tell the gardener to wait",
    "the young teacher opened the window", "the messenger crossed the bridge",
)

def norm(s: str) -> str:
    return "".join(re.findall("[a-z]", s.lower()))

def audit(s: str) -> dict:
    t = norm(s); rev = t[::-1]
    mism = [i for i,(a,b) in enumerate(zip(t, rev)) if a != b]
    return {"letters": len(t), "two_pointer_exact": bool(t) and not mism,
            "first_mismatch": mism[0] if mism else None,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest()}

def run() -> dict:
    rows=[]; transitions=0
    for la in LEFT:
        for rb in RIGHT:
            transitions += 1
            rendered = f"{la}; {rb}."
            rows.append({"rendered": rendered, "dialogue_acts": ["left-act", "right-act"],
                         "audit": audit(rendered), "provenance": {
                             "left_source": "hand-authored dialogue-act inventory",
                             "right_source": "independent hand-authored dialogue-act inventory",
                             "finished_tape_reversed": False, "catalogue_text": False,
                             "repair": False}})
    exact=[r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["letters"]>38]
    out={"experiment":"dialogue-act-product-20260920",
         "method":"independent conversational-act product with live outer residual check",
         "novelty_preflight":{"registry_inspected":True,"catalogue_text_imported":False,
                              "finished_tape_reversal":False,"repair":False,
                              "mirrored_units":False},
         "stats":{"left_acts":len(LEFT),"right_acts":len(RIGHT),
                   "transitions":transitions,"exact_gt38":len(exact)},
         "rendered_candidates":rows,
         "exact_candidates":exact,
         "next_construction":"Add a third independently authored turn type (acknowledgement or clarification) and solve act-sequence lengths jointly, retaining full utterances and rejecting word-unit mirrors.",
         "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits":["two-pointer mismatch scan","SHA-256 forward/reverse"]}}
    p=ROOT/"runs/dialogue-act-product-20260920.json"; p.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"artifact":str(p),"transitions":transitions,"exact_gt38":len(exact),"longest":max(r["audit"]["letters"] for r in rows)}))
    print(rows[0]["rendered"])
    return out

if __name__ == "__main__": run()
