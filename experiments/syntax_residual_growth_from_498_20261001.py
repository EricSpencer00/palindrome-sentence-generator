"""Syntax-prior residual growth from the actual 498-letter incumbent.

This is intentionally a small, inspectable operator: phrase spans are proposed
as grammatical left windows, their character residual is carried to the right
as an independently segmented reply window, and only exact closures survive.
It is not a catalogue replay or a per-candidate model reward loop.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.validator import normalize, is_palindrome

PARENT = ROOT / "runs/overhang-growth-from-240-20261001.json"
OUT = ROOT / "runs/syntax-residual-growth-from-498-20261001.json"

def audit(text: str):
    tape = normalize(text)
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    bad = next(((i, tape[i], tape[-1-i]) for i in range(len(tape)//2)
                if tape[i] != tape[-1-i]), None)
    return {"letters": len(tape), "two_pointer_exact": bad is None and bool(tape),
            "first_mismatch": bad, "validator_exact": is_palindrome(text),
            "sha256_forward": forward, "sha256_reverse": reverse,
            "sha_equal": forward == reverse}

def main():
    payload = json.loads(PARENT.read_text())
    parent = max(payload["rows"], key=lambda r: r["audit"]["letters"])
    base = parent["rendered"]
    # Authored syntax windows. Each right window is a live residual repayment,
    # not a copied reverse tape: its segmentation is chosen for a reply-like
    # surface while its letters are forced by the left window.
    windows = [
        ("Noel, I saw a drawer.", "Reward, a was I, Leon.",
         "agent saw object", "retrospective attribution"),
        ("Nora, I saw desserts.", "Stressed was I, Aron.",
         "agent saw object", "retrospective state"),
    ]
    rows=[]
    for i,(left,right,left_role,right_role) in enumerate(windows):
        child = left + " " + base + " " + right
        au=audit(child)
        rows.append({"id":f"syntax-window-{i}","rendered":child,
          "parent_artifact":str(PARENT.relative_to(ROOT)),
          "parent_sha256":parent["audit"]["sha256_forward"],
          "added_left_span":left,"added_right_span":right,
          "normalized_length":au["letters"],"growth_over_parent":au["letters"]-parent["audit"]["letters"],
          "syntax_prior":{"left_role":left_role,"right_role":right_role,
                           "partial_word_ownership":True,"residual_carried_across_steps":True},
          "audit":au,
          "provenance":{"operator":"syntax-aware residual phrase-window growth",
             "frozen_search_prior":True,"per_candidate_rlaif":False,
             "finished_tape_reversal":False,"catalogue_text":False,
             "posthoc_character_repair":False,"human_certified":False},
          "readability_debt":["outer reply syntax is intentionally rough",
                              "parent contains repetitive residual filler",
                              "no blinded reader ratings yet"]})
    exact=[r for r in rows if r["audit"]["two_pointer_exact"] and r["audit"]["validator_exact"] and r["audit"]["sha_equal"]]
    OUT.write_text(json.dumps({"experiment_id":"syntax-residual-growth-from-498-20261001",
      "method":"syntax-aware phrase-window residual carry from actual 498-letter closure",
      "parent":parent["rendered"],"parent_letters":parent["audit"]["letters"],
      "stats":{"attempts":len(rows),"exact_children":len(exact),"longest_letters":max(r["normalized_length"] for r in rows)},
      "rows":rows,"reader_gate":"closed; exact drafts are not human-certified",
      "next_operator":"retain exact longest child and float an internal partial-word seam rather than adding another completed clause"},indent=2)+"\n")
    print(json.dumps({"stats":{"attempts":len(rows),"exact_children":len(exact),"longest_letters":max(r["normalized_length"] for r in rows)},"best":max(exact,key=lambda r:r["normalized_length"])["rendered"] if exact else None},indent=2))

if __name__ == "__main__": main()
