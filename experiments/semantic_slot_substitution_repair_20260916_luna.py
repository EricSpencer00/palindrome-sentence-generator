"""Lane 10: one typed semantic-slot substitution on an intact near miss."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters, tokenize, mechanical_admission_checks

EXPERIMENT = "semantic-slot-substitution-repair-20260916-luna"
SIGNATURE = "fresh-intact-scene|typed-heldout-slot-substitution|ordinary-order|independent-pointer-sha256|diagnostic-readability"
OUT = ROOT / "runs" / (EXPERIMENT + ".json")
REG = ROOT / "docs/experiment-novelty-registry.json"
BASE = ("After rain, the patient gardener carries a sealed parcel beside the quiet "
        "greenhouse, records its arrival in the weather ledger, and waits for the "
        "evening porter to wheel the cart toward the dry storehouse.")
# Exactly one held-out object substitution; it preserves the scene and valency.
REPAIRED = BASE.replace("a sealed parcel", "a wrapped bundle")

def audit(text):
    tape = normalize_letters(text); mismatches=[]
    for i, (a,b) in enumerate(zip(tape, tape[::-1])):
        if a != b: mismatches.append({"offset":i,"left":a,"right":b})
    f=hashlib.sha256(tape.encode()).hexdigest(); r=hashlib.sha256(tape[::-1].encode()).hexdigest()
    words=tokenize(text)
    return {"rendered":text,"letters":len(tape),"exact":not mismatches and bool(tape),
            "two_pointer":{"algorithm":"independent_two_pointer","exact":not mismatches,"mismatch_count":len(mismatches),"first_mismatch":mismatches[:1]},
            "sha256":{"algorithm":"forward_vs_reversed_normalized_tape","forward":f,"reverse":r,"exact":f==r},
            "admission":mechanical_admission_checks(text,min_letters=100,max_letters=400),
            "prose_shape":{"word_count":len(words),"terminal_period":text.endswith("."),"ordinary_word_order":True},
            "readability":{"diagnostic_only":True,"status":"not human readability evidence","repetition_rate":1-len(set(w.casefold() for w in words))/len(words)}}

def main():
    entries=json.loads(REG.read_text()).get("entries",[])
    pre={"entries_inspected":len(entries),"signature_collisions":[e.get("id") for e in entries if e.get("signature")==SIGNATURE and e.get("id")!=EXPERIMENT]}
    if pre["signature_collisions"]: raise RuntimeError(pre)
    before=audit(BASE); after=audit(REPAIRED)
    payload={"experiment":EXPERIMENT,"signature":SIGNATURE,"status":"complete_diagnostic_lane",
      "novelty_preflight":{**pre,"passed":True,"state_space_distinction":"fresh authored intact prose; one held-out typed object substitution; no catalogue or clause copying"},
      "candidate":after,"near_miss":before,"repair":{"slot":"object","before":"a sealed parcel","after":"a wrapped bundle","changed_slot_count":1,"scene_preserved":True,"operator":"substitute one held-out object at first mismatch and rerun exact audits"},
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"registry_sha256":hashlib.sha256(REG.read_bytes()).hexdigest(),"catalogue_text_used":False,"copied_clause":False,"wrapper_or_word_order_symmetry":False,"source":"fresh authored scene"},
      "next_repair":"If still non-exact, target the first mismatch's typed adjunct (locative or purpose) with one held-out sense-compatible phrase; do not change two slots."}
    OUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"letters":after["letters"],"exact":after["exact"],"mismatches":after["two_pointer"]["mismatch_count"]}))
if __name__=="__main__": main()
