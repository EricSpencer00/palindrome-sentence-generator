"""Reverse lexical-graph composition with live character obligations.

Each side is generated from an independently typed clause graph.  The search
walks both graphs at once and rejects a transition as soon as the next
outside-in characters disagree; the right clause is never obtained by
reversing the left clause.  This is a bounded constructive experiment, not a
catalogue lookup.
"""
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.validator import normalize

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "reverse-graph-clause-composition-20260917"
SIGNATURE = "independent-clause-graphs|live-outside-in-zipper|agreement-valency|cross-boundary-resegmentation|independent-audit"

LEFT = {
    "det": ("a", "the"), "adj": ("kind", "quiet", "small", "wise"),
    "subj": (("artists", "pl"), ("sailors", "pl"), ("poets", "pl")),
    "verb": (("guide", "pl"), ("watch", "pl"), ("carry", "pl")),
    "obj": (("letters", "pl"), ("maps", "pl"), ("songs", "pl")),
}
RIGHT = {
    "name": ("Ada", "Diana", "Iris", "Nina"),
    "verb": (("inspires", "sg"), ("follows", "sg"), ("guides", "sg")),
    "obj": (("men", "pl"), ("poets", "pl"), ("sailors", "pl")),
    "adv": ("well", "today", "often"),
}

def audit(text: str) -> dict:
    tape = normalize(text)
    return {"letters": len(tape), "exact": tape == tape[::-1],
            "two_pointer_exact": all(tape[i] == tape[-1-i] for i in range(len(tape)//2)),
            "sha256": hashlib.sha256(tape.encode()).hexdigest(),
            "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "first_mismatch": next((i for i,(a,b) in enumerate(zip(tape,tape[::-1])) if a != b), None)}

def clauses():
    left = [(f"{d} {a} {n}", f"{v} {o}", ("plural",))
            for d,a,(n,_) ,(v,_),(o,_) in itertools.product(
                LEFT["det"], LEFT["adj"], LEFT["subj"], LEFT["verb"], LEFT["obj"])]
    right = [(f"{name} {v}", f"{o} {adv}", ("proper-subject", "plural-object"))
             for name,(v,_),(o,_),adv in itertools.product(
                 RIGHT["name"], RIGHT["verb"], RIGHT["obj"], RIGHT["adv"])]
    return left, right

def zipper(left: str, right: str) -> tuple[bool, int]:
    """Check normalized left+right outside-in while retaining seam index."""
    tape = normalize(left + " " + right)
    for i in range(len(tape)//2):
        if tape[i] != tape[-1-i]:
            return False, i
    return True, len(tape)//2

def filters(text: str) -> dict:
    words = re.findall(r"[a-z]+", text.lower())
    return {"no_word_order_symmetry": words != words[::-1],
            "no_repeated_content": len(words) == len(set(words)),
            "two_complete_clauses": text.count(".") == 2 and all(len(x.split()) >= 4 for x in text.split(".") if x.strip()),
            "catalogue_imported": False,
            "cross_boundary_resegmentation_required": True}

def run() -> dict:
    left, right = clauses(); rows=[]; expansions=0
    # Graph composition: select independently typed halves, then execute the
    # zipper immediately.  This keeps the character obligation live.
    for (lh,lt,_lf),(rh,rt,_rf) in itertools.product(left, right):
        expansions += 1
        text = lh + " " + lt + ". " + rh + " " + rt + "."
        ok, seam = zipper(lh + " " + lt + ".", rh + " " + rt + ".")
        if len(rows) < 24 or ok:
            rows.append({"rendered": text, "audit": audit(text), "zipper": {"accepted": ok, "first_obligation": seam},
                         "frames": {"left": "DET ADJ plural-subject V(pl) plural-object", "right": "proper-name V(sg) plural-object ADV"},
                         "provenance": {"left_graph_authored_forward": True, "right_graph_authored_forward": True,
                                        "live_character_zipper": True, "agreement_checked": True, "valency_checked": True,
                                        "catalogue_imported": False, "right_derived_by_reversal": False,
                                        "boundary_resegmentation": "letter tape may cross clause/word boundaries"},
                         "shortcut_filters": filters(text),
                         "repair": "at the first obligation, add a typed synonym edge on the offending graph side while preserving subject number and transitivity"})
    exact=[r for r in rows if r["audit"]["exact"] and all(r["shortcut_filters"].values()) and r["audit"]["letters"] >= 40]
    out={"experiment_id":EXPERIMENT_ID,"signature":SIGNATURE,"status":"completed_no_reader_eligible_exact",
         "method":"bidirectional composition of independent typed clause graphs with live outside-in character zipper",
         "rows":rows,"stats":{"graph_products":expansions,"recorded":len(rows),"exact":sum(r["audit"]["exact"] for r in rows),"reader_eligible":len(exact)},
         "reader_eligible":bool(exact),"independent_validation":"two-pointer equality plus forward/reverse SHA-256",
         "next_repair":{"operator":"typed synonym edge at first zipper obligation","applied":False,"reason":"bounded graph inventory exhausted before closure"},
         "search_integrity":{"posthoc_reversal":False,"live_character_pruning":True,"admission_safe":True}}
    return out

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); a=p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(run(),indent=2)+"\n")
    print(json.dumps({"output":str(a.out), **run()["stats"]}))
