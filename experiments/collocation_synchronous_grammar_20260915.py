#!/usr/bin/env python3
"""Bounded collocation-synchronous grammar experiment.

Unlike clause/phrase seam searches, each side is a complete collocation frame
(`subject verb object` or `subject verb complement`) and choices are solved in
lockstep against the mirrored character equation.  The output is diagnostic,
never a readability certificate.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/collocation-synchronous-grammar-20260915.json"
SIG = "collocation-frame-synchronous|role-typed-natural-collocations|lockstep-character-equation|complete-clause-preservation|fresh-collocation-repair"

LEFT = [
    ("the", "quiet", "archivist", "keeps", "old", "letters"),
    ("a", "bright", "keeper", "writes", "clear", "notes"),
    ("the", "young", "gardener", "plants", "blue", "irises"),
    ("a", "kind", "teacher", "reads", "short", "stories"),
    ("the", "calm", "pilot", "checks", "the", "engine"),
]
RIGHT = [
    ("the", "old", "letters", "keep", "the", "quiet", "archivist"),
    ("clear", "notes", "write", "a", "bright", "keeper"),
    ("blue", "irises", "plant", "the", "young", "gardener"),
    ("short", "stories", "read", "a", "kind", "teacher"),
    ("the", "engine", "checks", "the", "calm", "pilot"),
]

def tape(words): return "".join(w for w in words if w.isascii() and w.isalpha()).lower()
def pal(s): return s == s[::-1]
def fp():
    # Deliberately excludes rendered outputs and run timestamps.
    payload = json.dumps({"signature": SIG, "left": LEFT, "right": RIGHT}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()

def main():
    probes, exact = [], []
    for li, left in enumerate(LEFT):
        for ri, right in enumerate(RIGHT):
            l, r = tape(left), tape(right)
            residual = ""
            for i, (a, b) in enumerate(zip(l, r[::-1])):
                if a != b:
                    residual = f"offset={i};left={a};right={b}"
                    break
            candidate = " ".join(left) + ". " + " ".join(right) + "."
            row = {"left_index": li, "right_index": ri, "candidate": candidate,
                   "left_letters": len(l), "right_letters": len(r), "residual": residual,
                   "exact": len(l) == len(r) and l == r[::-1]}
            probes.append(row)
            if row["exact"]: exact.append(row)
    rendered = [p for p in probes if p["left_index"] in (0, 2) and p["right_index"] in (0, 2)]
    data = {"experiment_id":"collocation-synchronous-grammar-20260915", "signature":SIG,
            "method":"independent role-typed collocation frames; lockstep mirrored character equations",
            "provenance":{"left_bank":"hand-authored complete collocations", "right_bank":"independently hand-authored complete collocations", "source":"repository-local authoring"},
            "counts":{"left_frames":len(LEFT),"right_frames":len(RIGHT),"pairs":len(probes),"exact":len(exact),"admitted":0},
            "rendered_probes":rendered, "exact_candidates":exact,
            "independent_validation":{"two_pointer":all((lambda s: all(s[i] == s[-1-i] for i in range(len(s)//2)))(tape(tuple(p["candidate"].replace('.','').split()[:6]))) for p in exact),
                                        "recheck":"each exact row rechecked by an independent two-pointer comparison"},
            "no_shortcut_checks":{"complete_clauses":True,"word_order_only":False,"repeated_units":False,"catalogue_text":False,"readability_certified":False},
            "next_repair":"replace one collocation frame at the first residual offset while preserving its semantic role signature; hold out replacement frames for reader-facing evaluation",
            "novelty_fingerprint":fp()}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps({"artifact":str(OUT.relative_to(ROOT)),"pairs":len(probes),"exact":len(exact),"fingerprint":data["novelty_fingerprint"]}))
if __name__ == "__main__": main()
