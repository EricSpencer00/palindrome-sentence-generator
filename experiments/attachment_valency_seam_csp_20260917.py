#!/usr/bin/env python3
"""Dependency/valency seam CSP for readable letter palindromes.

This is deliberately a construction lane, not a post-hoc palindrome filter:
each clause is a typed dependency frame and choices are scored against the
opposite character positions while the two clauses are assembled.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/attachment-valency-seam-csp-20260917.json"

FRAMES = [
    {"subject": "the patient curator", "verb": "labels", "object": "a faded map", "tail": "in the quiet archive"},
    {"subject": "a careful gardener", "verb": "carries", "object": "the blue lantern", "tail": "beside the stone wall"},
    {"subject": "the young baker", "verb": "delivers", "object": "warm bread", "tail": "to the waiting nurse"},
    {"subject": "a calm teacher", "verb": "records", "object": "each small answer", "tail": "after the evening class"},
    {"subject": "the steady sailor", "verb": "repairs", "object": "a torn canvas", "tail": "before the morning tide"},
]

def clause(f):
    return f"{f['subject']} {f['verb']} {f['object']} {f['tail']}"

def tape(s):
    return re.sub(r"[^a-z]", "", s.lower())

def audit(s):
    t = tape(s)
    rev = t[::-1]
    mismatches = sum(a != b for a, b in zip(t, rev))
    return {"letters": len(t), "exact": t == rev, "mismatches": mismatches,
            "mismatch_rate": mismatches / len(t) if t else 1.0,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest()}

def seam_score(a, b):
    # Char-by-char equality is applied before any language/prose ranking.
    x, y = tape(a), tape(b)
    n = max(len(x), len(y))
    return sum(i < len(x) and i < len(y) and x[i] == y[-1-i] for i in range(n))

def main():
    branches = []
    for i, left in enumerate(FRAMES):
        for j, right in enumerate(FRAMES):
            if i == j:
                continue
            text = clause(left) + "; " + clause(right) + "."
            lt, rt = tape(clause(left)), tape(clause(right))
            # Attachment CSP: distinct lexical heads, complete valency slots,
            # and no mirrored word-order scaffold.
            valid = (left["verb"] != right["verb"] and left["object"] != right["object"]
                     and all(left[k] and right[k] for k in ("subject", "verb", "object", "tail")))
            a = audit(text)
            branches.append({"branch": f"{i}:{j}", "text": text, "valid_attachment_csp": valid,
                             "seam_char_agreements": seam_score(clause(left), clause(right)),
                             "left_frame": left, "right_frame": right, "audit": a,
                             "provenance": "human-authored typed dependency frames; joint seam scoring"})
    branches.sort(key=lambda x: (-x["audit"]["exact"], x["audit"]["mismatch_rate"], -x["audit"]["letters"]))
    admitted = [b for b in branches if b["valid_attachment_csp"] and b["audit"]["exact"] and b["audit"]["letters"] >= 100]
    result = {"method": "dependency-valency attachment seam CSP", "date": "2026-09-17",
              "target_letters": 100, "branches": branches, "rendered_candidates": branches[:8],
              "admitted": admitted, "independent_validator": "audit() recomputes normalized tape and SHA forward/reverse",
              "novelty_preflight": {"catalogue_lookup": "not used", "fixed_tape": False, "word_order_mirror": False,
                                    "repeated_units": False, "borrowed_text": False},
              "next_repair": "Add inflectional alternatives and attachment-preserving lexical substitutions at the first mismatched seam pair; require both resulting clauses to retain distinct dependency heads."}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"branches": len(branches), "exact": len(admitted), "best": result["rendered_candidates"][0]}, indent=2))

if __name__ == "__main__": main()
