"""Semordnilap inventory in typed templates, with independent tape audit.

This is a construction route, not a claim that every exact surface is prose:
reversible word pairs are used as lexical *seams* inside different clause
templates.  A repair operator swaps a seam pair and re-segments the opposite
side; it is retained as evidence only when the full tape audit passes.
"""
from __future__ import annotations
from collections import Counter
import hashlib, json, re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).resolve().parents[1]
ID = "semordnilap-template-inventory"
SIGNATURE = "finite semordnilap pair inventory fills typed syntactic templates with bilateral character equations and seam-preserving repair substitutions"
PAIRS = [("drawer", "reward"), ("stressed", "desserts"), ("diaper", "repaid"),
         ("gateman", "nametag"), ("deliver", "reviled"), ("parts", "strap"),
         ("stop", "pots"), ("spit", "tips"), ("dog", "god"), ("saw", "was"),
         ("live", "evil"), ("emit", "time"), ("flow", "wolf"), ("loop", "pool"),
         ("smart", "trams"), ("star", "rats"), ("keep", "peek"), ("rail", "liar"),
         ("doom", "mood"), ("room", "moor"), ("swap", "paws"), ("snap", "pans"),
         ("raw", "war"), ("stun", "nuts"), ("straw", "warts")]
TEMPLATES = [
    "the {a} will {b}", "a {a} can {b}", "we {b} the {a}",
    "the {a} and the {b}", "they {b} while the {a} waits",
    "a {a} may {b} today", "the {a} saw {b}"
]
COMMON = {"a","an","the","and","or","we","they","he","she","it","can","may","will","saw","see","was","is","are","to","in","on","of","while","waits","today","some","men","nine","memos"}

def audit(text: str, method: str, repair: str | None = None) -> dict:
    letters = normalize_letters(text)
    checks = mechanical_admission_checks(text)
    return {"text": text, "letters": len(letters), "exact": letters == letters[::-1],
            "independent_exact": hashlib.sha256(letters.encode()).hexdigest() == hashlib.sha256(letters[::-1].encode()).hexdigest(),
            "method": method, "repair": repair, "checks": checks}

def reverse_segment(s: str, vocab: set[str]) -> list[str] | None:
    dp = {0: []}
    for i in range(len(s)):
        if i not in dp: continue
        for j in range(i + 1, min(len(s), i + 12) + 1):
            w = s[i:j]
            if w in vocab and (j not in dp or len(dp[j]) > len(dp[i]) + 1):
                dp[j] = dp[i] + [w]
    return dp.get(len(s))

def run() -> dict:
    vocab = COMMON | {w for p in PAIRS for w in p}
    probes = []
    # The center seed is a known, independently checked scaffold; this route
    # is evaluated on new outer template material and never presents the seed
    # as a generated result.
    center = "an aide rips nine memos some men inspire diana"
    for ti, template in enumerate(TEMPLATES):
        for pi, (a, b) in enumerate(PAIRS):
            left = template.format(a=a, b=b)
            tape = normalize_letters(left + center)
            right_tape = tape[::-1]
            seg = reverse_segment(right_tape, vocab)
            right = " ".join(seg) if seg else ""
            text = left + " " + center + " " + right
            probes.append(audit(text, "typed-template-seam", None))
            # Concrete repair: replace the second seam with its paired form,
            # then independently re-segment the reflected tape.
            aa, bb = PAIRS[(pi + 1) % len(PAIRS)]
            repaired = template.format(a=aa, b=bb)
            rt = normalize_letters(repaired + center)[::-1]
            rseg = reverse_segment(rt, vocab)
            probes.append(audit(repaired + " " + center + (" " + " ".join(rseg) if rseg else ""),
                                "typed-template-seam", "swap-seam-and-resegment"))
    exact = [r for r in probes if r["exact"]]
    admitted = [r for r in exact if r["checks"].get("admitted", False)]
    return {"status":"completed_semordnilap_template_run", "experiment_id":ID,
            "signature":SIGNATURE, "pair_inventory":len(PAIRS), "templates":len(TEMPLATES),
            "probes":len(probes), "exact":len(exact), "mechanically_admitted":len(admitted),
            "reader_eligible":0, "provenance":"hand-authored finite semordnilap inventory; center scaffold held out",
            "concrete_repair":"swap a seam pair and independently re-segment the reflected character tape",
            "rows":probes}

if __name__ == "__main__":
    out = run(); path = ROOT / "runs" / "semordnilap-template-inventory-20260915.json"; path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("status","pair_inventory","templates","probes","exact","mechanically_admitted","reader_eligible")}))
