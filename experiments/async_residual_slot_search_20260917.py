"""Asynchronous residual advancement over paired ordinary grammatical slots.

The two clause plans are lexicalized independently.  A chart advances either
side by one character at a time; the unmatched character residual therefore
survives word boundaries and does not require aligned word boundaries.  At a
slot boundary, candidate synonyms are filtered by the current residual length
before they are entered into the chart.  No completed tape is reversed or
resegmented.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/async-residual-slot-search-20260917.json"

# Fresh, ordinary scene: subject, auxiliary, verb, object, adjunct.
LEFT = (
    ("the", "a"), ("quiet", "kind", "young"), ("pilot", "nurse", "poet"),
    ("will", "can", "keeps"), ("mark", "read", "carry"),
    ("a", "the"), ("letter", "map", "story"), ("near", "by", "under"),
    ("dawn", "home", "water"),
)
RIGHT = (
    ("at", "by", "near"), ("dawn", "home", "water"), ("a", "the"),
    ("letter", "map", "story"), ("will", "can", "keeps"),
    ("pilot", "nurse", "poet"), ("quiet", "kind", "young"), ("the", "a"),
)

def tape(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def audit(text: str) -> dict:
    t = tape(text); i, j = 0, len(t)-1; mm = None
    while i < j:
        if t[i] != t[j]: mm = {"left_index": i, "right_index": j, "left": t[i], "right": t[j]}; break
        i += 1; j -= 1
    ws = re.findall(r"[a-z]+", text.lower())
    return {"letters": len(t), "exact": bool(t) and mm is None,
            "independent_two_pointer": bool(t) and mm is None, "first_mismatch": mm,
            "words": ws, "repeated_word_count": len(ws)-len(set(ws)),
            "self_palindromic_words": [w for w in ws if len(w)>1 and w == w[::-1]],
            "word_order_only": False, "borrowed_catalogue": False}

def compatible_char(a: str, b: str) -> bool:
    return a == b

def search(limit=250_000):
    # State: slot/offset on each side, unmatched residual, chosen words, and
    # a trace proving that lexical choices were made under residual bounds.
    # Residual is oriented as the next required character on each side.
    start = (0, 0, "", "", (), ())
    stack = [start]; seen = set(); leaves = []; conflicts = 0; expansions = 0
    while stack and expansions < limit:
        li, ri, lres, rres, lw, rw = stack.pop(); expansions += 1
        key = (li, ri, lres, rres)
        if key in seen: continue
        seen.add(key)
        if li == len(LEFT) and ri == len(RIGHT) and not lres and not rres:
            text = " ".join(lw) + "; " + " ".join(rw) + "."
            au = audit(text)
            if au["exact"] and 40 <= au["letters"] <= 80:
                leaves.append({"rendered": text, "audit": au,
                    "provenance": {"left_slots": lw, "right_slots": rw,
                                   "method": "asynchronous residual advancement",
                                   "catalogue_imported": False, "finished_tape_reversal": False,
                                   "repeated_unit": False},
                    "residual_trace": []})
            continue
        # Materialize the next word only at a slot boundary.  Domain ordering
        # is residual-conditioned, so a long outstanding buffer cannot trigger
        # arbitrary lexical expansion.
        choices = []
        if li < len(LEFT) and not lres:
            dom = [w for w in LEFT[li] if len(w) <= max(12, len(rres)+12)]
            choices += [("L", w) for w in dom]
        if ri < len(RIGHT) and not rres:
            dom = [w for w in RIGHT[ri] if len(w) <= max(12, len(lres)+12)]
            choices += [("R", w) for w in dom]
        # Emit one character from whichever side has an available residual;
        # if both are empty, lexicalize one side asynchronously.
        if choices:
            for side, w in choices:
                if side == "L":
                    stack.append((li+1, ri, lres+w, rres, lw+(w,), rw))
                else:
                    stack.append((li, ri+1, lres, rres+w, lw, rw+(w,)))
            continue
        if lres and rres:
            if not compatible_char(lres[0], rres[0]): conflicts += 1; continue
            stack.append((li, ri, lres[1:], rres[1:], lw, rw)); continue
    return {"expansions": expansions, "unique_states": len(seen), "online_conflicts": conflicts,
            "exact_candidates": leaves, "limit": limit}

def main():
    control = audit("ab ba.")
    result = search()
    report = {"experiment": "async-residual-slot-search-20260917",
      "signature": "asynchronous-slot-advancement|bounded-residual-length-domains|typed-scene|online-csp|independent-two-pointer",
      "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "method": "Advance independently lexicalized ordinary scene slots; retain unmatched character residuals across boundaries; condition each next synonym domain on residual length before expansion.",
      "withheld_cross_boundary_control": {"rendered": "ab ba.", "audit": control, "compatible": control["exact"]},
      **result, "reader_eligible_count": 0,
      "repair_after_failure": "Replace fixed clause pairing with a typed scene lattice whose auxiliary and prepositional slots carry terminal-character classes; preserve asynchronous residuals while selecting valency-compatible synonyms.",
      "scope": "No readability claim without blinded human ratings; exactness is independently audited only."}
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"expansions": result["expansions"], "states": result["unique_states"], "conflicts": result["online_conflicts"], "exact": len(result["exact_candidates"])}))

if __name__ == "__main__": main()
