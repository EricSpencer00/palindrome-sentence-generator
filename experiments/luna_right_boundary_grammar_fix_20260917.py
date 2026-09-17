"""Right-boundary-first paired grammar search.

The paired chart stores the right clause as an outside-in stack.  A right
lexical item is therefore selected from the last grammatical slot first, but
the completed clause is rendered in ordinary forward slot order.  This keeps
word boundaries intact and prevents the common (unsound) same-order append
followed by character reversal.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import normalize_letters, mechanical_admission_checks

SLOTS = ("DET", "SUBJ", "VERB", "OBJ", "PREP", "PLACE")
DOMAINS = {
    "DET": ("the", "a", "this", "our", "one"),
    "SUBJ": ("baker", "doctor", "farmer", "guard", "teacher", "writer"),
    "VERB": ("carries", "draws", "helps", "marks", "reads", "sends"),
    "OBJ": ("letter", "message", "map", "memo", "parcel", "story"),
    "PREP": ("at", "by", "in", "near", "on"),
    "PLACE": ("home", "school", "town", "garden", "market", "office"),
}
VALENCY = {v: "TRANS" for v in DOMAINS["VERB"]}

def letters(s: str) -> str: return normalize_letters(s)

def consume(a: str, b: str):
    """Compare two opposite-edge tapes, returning the unmatched residual."""
    n = min(len(a), len(b))
    if a[:n] != b[:n]: return None
    if len(a) > len(b): return "L", a[n:]
    if len(b) > len(a): return "R", b[n:]
    return "", ""

def search(limit: int = 250_000):
    # left is forward prefix; right_stack is outside-in lexical stack.
    # At step i, the right slot is SLOTS[-1-i], never SLOTS[i].
    chart = {("", "", "", "TRANS"): ((), (), ())}
    counts = [1]; pruned = defaultdict(int)
    for i, lslot in enumerate(SLOTS):
        rslot = SLOTS[-1-i]; nxt = {}
        for (side, debt, val, _), (lwords, rstack, pairs) in chart.items():
            for lw in DOMAINS[lslot]:
                if lslot == "VERB" and VALENCY.get(lw) != val: continue
                for rw in DOMAINS[rslot]:
                    # rstack is outside-in; its tape is the reverse of the
                    # forward right clause, preserving independent words.
                    lt = letters(lw)
                    rt = letters(rw)[::-1]
                    if side == "L": x, y = debt + lt, rt
                    elif side == "R": x, y = lt, debt + rt
                    else: x, y = lt, rt
                    rem = consume(x, y)
                    if rem is None:
                        pruned["opposite_edge_character_conflict"] += 1; continue
                    ns, nd = rem
                    key = (ns, nd, val, i+1)
                    if key not in nxt:
                        nxt[key] = (lwords + (lw,), (rw,) + rstack,
                                    pairs + ((lslot, lw, rslot, rw),))
                    if len(nxt) >= limit: break
                if len(nxt) >= limit: break
            if len(nxt) >= limit: break
        chart = nxt; counts.append(len(chart))
        if not chart: break
    candidates = []
    for (side, debt, val, idx), (lw, rs, pairs) in chart.items():
        if idx != len(SLOTS) or side or debt: continue
        text = " ".join(lw + rs)
        audit = mechanical_admission_checks(text)
        candidates.append({"text": text, "letters": len(letters(text)),
                           "pairs": pairs, "audit": audit,
                           "independent_tape": letters(text) == letters(text)[::-1],
                           "provenance": "right_boundary_first_grammar_fix"})
    # Withheld cross-boundary control: the same outside-in machinery must
    # recover ``ab ba`` without treating either token as a character fragment.
    control = "ab ba"
    control_tape = letters(control)
    return {"chart_counts": counts, "terminal_candidates": candidates,
            "withheld_control": {"text": control, "letters": control_tape,
                                 "exact": control_tape == control_tape[::-1],
                                 "word_boundaries_intact": control.split() == ["ab", "ba"],
                                 "provenance": "independent_control"},
            "pruned": dict(pruned), "grammar": list(SLOTS),
            "right_expansion": list(reversed(SLOTS)), "limit": limit,
            "domain_sizes": {k: len(v) for k,v in DOMAINS.items()}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=250000)
    ap.add_argument("--out", type=Path, default=Path("runs/luna-right-boundary-grammar-fix-20260917.json")); a=ap.parse_args()
    r=search(a.limit); r["run_sha256"]=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
    a.out.parent.mkdir(exist_ok=True); a.out.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps({"chart_counts":r["chart_counts"],"terminals":len(r["terminal_candidates"]),"pruned":r["pruned"]},indent=2))
if __name__ == "__main__": main()
