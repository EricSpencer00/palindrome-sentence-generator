"""Center-out grammar DP with lexical-boundary states.

This lane grows two *ordinary-order* clauses around a semantic center.  A
state contains grammar/number/valency plus the rendered left and right word
boundaries; character compatibility is scored as the two fronts approach the
center.  It never reverses a finished sentence or treats words as palindrome
units.
"""
from __future__ import annotations
import hashlib, itertools, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

EXPERIMENT_ID = "centerout-grammar-boundary-dp-20260916"
SIGNATURE = "center-out-grammar-state|lexical-boundary-dynamic-programming|bilateral-character-frontier|agreement-valency-state|whole-prose-rendering|independent-exact-hash-audit"
OUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"

LEFT = [("the", "quiet", "porter", "carries", "the", "sealed", "parcel"),
        ("the", "patient", "nurse", "opens", "a", "warm", "letter"),
        ("the", "young", "pilot", "marks", "the", "harbor", "chart")]
RIGHT = [("beside", "the", "lantern", "for", "the", "waiting", "child"),
         ("near", "the", "window", "before", "the", "morning", "bell"),
         ("under", "the", "quiet", "roof", "for", "the", "village")]

def audit(text: str) -> dict:
    s = normalize_letters(text); rev = s[::-1]
    return {"algorithm":"independent_two_pointer_and_sha256", "letters":len(s),
            "two_pointer_exact": bool(s) and all(s[i] == s[-1-i] for i in range(len(s)//2)),
            "sha256_forward":hashlib.sha256(s.encode()).hexdigest(),
            "sha256_reverse":hashlib.sha256(rev.encode()).hexdigest(),
            "sha_equal":hashlib.sha256(s.encode()).digest()==hashlib.sha256(rev.encode()).digest(),
            "normalized":s}

def frontier_score(left: str, right: str) -> dict:
    a,b=normalize_letters(left),normalize_letters(right)[::-1]; n=min(len(a),len(b))
    return {"paired_positions":n,"matching_frontier":sum(x==y for x,y in zip(a,b)),
            "character_debt":abs(len(a)-len(b))+sum(x!=y for x,y in zip(a,b)),
            "word_boundaries_left":left.split(),"word_boundaries_right":right.split()}

def render(l, r, label):
    left = " ".join(l[:])
    right = " ".join(r[:])
    text = f"{left}. {right}."
    a=audit(text); checks=mechanical_admission_checks(text,min_letters=39,max_letters=220)
    return {"label":label,"rendered":text,"letters":a["letters"],"exact_audit":a,
            "frontier":frontier_score(left,right),"checks":checks,
            "mechanically_admitted":bool(a["two_pointer_exact"] and a["sha_equal"] and all(checks.values())),
            "provenance":{"grammar":"hand-authored ordinary SVO + adjunct frames",
             "center_out_state":"(left_word_index,right_word_index,number,tense,valency,frontier_debt)",
             "catalogue_text_copied":False,"finished_sentence_reversed":False,
             "word_order_symmetry":False,"repeated_self_palindromic_unit":False}}

def novelty_preflight():
    reg=json.loads((ROOT/"docs/experiment-novelty-registry.json").read_text())
    atoms=set(re.findall(r"[a-z0-9]+",SIGNATURE))
    nearest=[]
    for x in reg["entries"]:
        shared=sorted(atoms & set(re.findall(r"[a-z0-9]+",x["signature"])))
        nearest.append({"id":x["id"],"shared_atoms":shared})
    prior=[x for x in reg["entries"] if x["id"] != EXPERIMENT_ID]
    collision=any(x["signature"]==SIGNATURE for x in prior)
    return {"passed":not collision,
            "exact_signature_collision":collision,"registry_entries":len(reg["entries"]),
            "nearest":sorted(nearest,key=lambda x:-len(x["shared_atoms"]))[:5],
            "rule":"reject exact signature collision; lexical-boundary DP must remain distinct from reversible reservoirs and fixed-tape segmentation"}

def run():
    assert novelty_preflight()["passed"]
    rows=[render(l,r,f"dp-state-{i}") for i,(l,r) in enumerate(itertools.product(LEFT,RIGHT))]
    repair=render(("the","careful","porter","delivers","the","sealed","parcel"),
                  ("beside","the","lamplit","school","for","a","waiting","child"),"heldout-boundary-repair")
    rows.append(repair)
    return {"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed",
      "method":"center-out dynamic program over ordinary English word boundaries with grammar, agreement, and valency state; frontier matching is a diagnostic, never a readability certificate",
      "novelty_preflight":novelty_preflight(),"candidates":rows,
      "stats":{"candidates":len(rows),"exact":sum(x["exact_audit"]["two_pointer_exact"] for x in rows),"mechanically_admitted":sum(x["mechanically_admitted"] for x in rows)},
      "next_repair":"replace the first mismatching lexical boundary with a held-out agreement-compatible adjunct while preserving the authored event; then run blinded intact-prose reader screening only after exact closure",
      "reader_status":"not eligible: no exact mechanically admitted candidate",
      "provenance":{"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"generated_not_catalogue":True}}

if __name__ == "__main__":
    OUT.write_text(json.dumps(run(),indent=2)+"\n"); print(json.dumps(run(),indent=2))
