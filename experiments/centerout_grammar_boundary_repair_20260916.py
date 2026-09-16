"""One held-out lexical-boundary repair for the center-out grammar DP lane.

The parent run identified the first mirrored character mismatch.  This repair
changes only the right adjunct, keeping the porter/parcel event and grammar
state fixed; it is deliberately not a replacement sweep.
"""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "centerout-grammar-boundary-repair-20260916"
SIGNATURE = "center-out-grammar-state|first-mismatch-boundary-repair|heldout-adjunct-substitution|event-preserving-valency|independent-exact-hash-audit"
OUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
PARENT = ROOT / "runs" / "centerout-grammar-boundary-dp-20260916.json"

def exact_audit(text: str) -> dict:
    s = normalize_letters(text); rev = s[::-1]
    mismatches=[]; i=0; j=len(s)-1
    while i<j:
        if s[i]!=s[j]: mismatches.append({"left":i,"right":j,"a":s[i],"b":s[j]})
        i+=1; j-=1
    hf=hashlib.sha256(s.encode()).hexdigest(); hr=hashlib.sha256(rev.encode()).hexdigest()
    return {"algorithm":"independent_two_pointer_plus_sha256","letters":len(s),
            "two_pointer_exact":bool(s) and not mismatches,"mismatch_count":len(mismatches),
            "first_mismatch":mismatches[0] if mismatches else None,
            "sha256_forward":hf,"sha256_reverse":hr,"sha_equal":hf==hr}

def run():
    parent=json.loads(PARENT.read_text())
    base=parent["candidates"][-1]
    # The event is held fixed. Only the adjunct boundary is replaced.
    old="beside the lamplit school for a waiting child"
    new="near the lamplit school for a waiting child"
    assert old in base["rendered"]
    rendered=base["rendered"].replace(old,new,1)
    audit=exact_audit(rendered)
    checks=mechanical_admission_checks(rendered,min_letters=39,max_letters=220)
    row={"label":"heldout-first-boundary-adjunct-repair","rendered":rendered,
         "letters":audit["letters"],"exact_audit":audit,"checks":checks,
         "mechanically_admitted":bool(audit["two_pointer_exact"] and audit["sha_equal"] and all(checks.values())),
         "repair":{"parent_label":base["label"],"changed_span":{"from":old,"to":new},
                   "changed_component":"right adjunct only","first_mismatch_target":base["exact_audit"].get("first_mismatch")},
         "provenance":{"authored_event_preserved":True,"source_parent":str(PARENT.relative_to(ROOT)),
             "catalogue_text_copied":False,"finished_sentence_reversed":False,
             "word_order_symmetry":False,"repeated_self_palindromic_unit":False}}
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed",
            "method":"single held-out adjunct substitution at the first lexical-boundary repair site; all other grammar and event slots fixed",
            "parent_run":str(PARENT.relative_to(ROOT)),"candidate":row,
            "stats":{"candidates":1,"exact":int(audit["two_pointer_exact"]),"mechanically_admitted":int(row["mechanically_admitted"])},
            "next_repair":"author one semantically compatible adjunct with the required boundary letters, then rerun the same independent exact gate; do not sweep substitutions",
            "reader_status":"not eligible: repaired candidate is not an exact mechanical closure",
            "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"generated_not_catalogue":True}}

if __name__ == "__main__":
    payload=run(); OUT.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))
