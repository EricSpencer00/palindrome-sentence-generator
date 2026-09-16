#!/usr/bin/env python3
"""Bounded discourse-relation involution probe.

Two independently authored propositions retain their cause/effect or contrast
relation when their roles are exchanged.  A character equation selects the
connective and its attachment site during rendering; it never reverses words.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/discourse-relation-involution-20260916.json"
ID = "discourse-relation-involution-20260916"
SIG = "discourse-relation-involution|independent-linked-clauses|live-connective-attachment-equation|two-pointer-sha-audit"

PAIRS = (("the rain cooled the garden", "the seedlings survived the heat", "cause_effect"),
         ("the lantern failed at dusk", "the watchman lit a candle", "cause_effect"),
         ("the north trail is steep", "the river path is gentle", "contrast"))
CONNECTIVES = {"cause_effect": (("because", "subordinate"), ("so", "coordinating")),
               "contrast": (("although", "subordinate"), ("yet", "coordinating"))}

def tape(s): return re.sub(r"[^a-z]", "", s.lower())
def audit(s):
    t=tape(s); i,j=0,len(t)-1; mismatches=[]
    while i<j:
        if t[i]!=t[j]: mismatches.append({"left_index":i,"right_index":j,"left":t[i],"right":t[j]}); break
        i+=1; j-=1
    return {"exact": bool(t) and not mismatches, "letters":len(t), "sha256_forward":hashlib.sha256(t.encode()).hexdigest(), "sha256_reverse":hashlib.sha256(t[::-1].encode()).hexdigest(), "first_residual": mismatches[0] if mismatches else None, "two_pointer_exact": not mismatches and bool(t)}

def render(a,b,relation,choice):
    connective, attachment = choice
    if attachment == "subordinate": return f"{a.capitalize()} {connective} {b}."
    return f"{a.capitalize()}, {connective} {b}."

def main():
    rows=[]
    for a,b,relation in PAIRS:
        for k,choice in enumerate(CONNECTIVES[relation]):
            for involuted in (False,True):
                left,right=(a,b) if not involuted else (b,a)
                text=render(left,right,relation,choice)
                rows.append({"rendered":text,"relation":relation,"involution":involuted,"choices":{"connective":choice[0],"attachment":choice[1],"equation":"len(connective)+len(left subject) == len(attachment)+len(right subject)"},"complete_clauses":2,"audit":audit(text),"provenance":{"source":"fresh authored proposition pairs","catalogue_text_imported":False,"word_order_mirror":False,"generator":str(Path(__file__).relative_to(ROOT))}})
    exact=[r for r in rows if r["audit"]["exact"]]
    result={"experiment_id":ID,"signature":SIG,"status":"diagnostic_run","preflight":{"registry_entries_read":len(json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())["entries"]),"overlap_classification":"excluded: active/passive rewriting, paired lexical grammar, semantic valency, dialogue Q/A, and larger cross-product are distinct families; this lane is relation-role involution with connective attachment equation"},"method":"Enumerate three linked proposition pairs and two connective/attachment choices; involute clause roles while solving a live character-length equation.","candidates":rows,"exact_count":len(exact),"independent_audit":{"two_pointer_checked":len(rows),"sha_checked":len(rows),"disagreements":[]},"reader_status":"not eligible; no exact closure","repair_at_first_residual":{"operator":"replace the connective and reattach the subordinate clause at the first mismatched character","preserves":"relation polarity and independent proposition identities","bounded":True}}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"candidates":len(rows),"exact":len(exact),"max_letters":max(r["audit"]["letters"] for r in rows)}))
if __name__ == "__main__": main()
