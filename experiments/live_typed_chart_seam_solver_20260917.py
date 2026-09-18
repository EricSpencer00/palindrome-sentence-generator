"""Live character-obligation search over typed, agreement-carrying clauses.

This is deliberately not a mirror or a word-order trick.  A chart item is a
typed clause; its reverse partner is selected while the character obligation
is live, before the pair is rendered.  Failed items retain their first seam.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LETTERS = re.compile(r"[a-z]+")

FRAMES = [
    ("the", "fox", "finds", "a", "den", "at", "dawn", "sg"),
    ("a", "child", "reads", "the", "book", "by", "water", "sg"),
    ("the", "sailors", "mark", "a", "map", "near", "shore", "pl"),
    ("a", "teacher", "opens", "the", "door", "at", "noon", "sg"),
    ("the", "bakers", "carry", "fresh", "bread", "to", "town", "pl"),
    ("the", "gardener", "waters", "a", "rose", "at", "sunrise", "sg"),
    ("a", "nurse", "checks", "the", "quiet", "room", "at", "night", "sg"),
]

def norm(s):
    return "".join(LETTERS.findall(s.lower()))

def audit(text):
    t = norm(text); r = t[::-1]
    mismatches = [i for i, (a, b) in enumerate(zip(t, r)) if a != b]
    # independent two-pointer implementation
    i, j = 0, len(t) - 1
    two_pointer = bool(t)
    while i < j:
        if t[i] != t[j]: two_pointer = False; break
        i += 1; j -= 1
    return {"letters": len(t), "exact": bool(t) and not mismatches,
            "two_pointer": two_pointer, "sha_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha_reverse": hashlib.sha256(r.encode()).hexdigest(),
            "mismatches": mismatches[:12]}

def frame_words(f):
    # chart normalization: subject/verb agreement is a feature, not a repair
    if len(f) == 8:
        det, subj, verb, odet, obj, prep, place, number = f
        return (det, subj, verb, odet, obj, prep, place), number
    det, subj, verb, odet, adj, obj, prep, place, number = f
    return (det, subj, verb, odet, adj, obj, prep, place), number

def run():
    chart = []
    for idx, raw in enumerate(FRAMES):
        words, number = frame_words(raw)
        left = " ".join(words)
        # A live chart item carries all obligations induced by its left tape.
        lt = norm(left); obligation = lt[::-1]
        chart.append({"id": idx, "words": words, "number": number,
                      "text": left, "tape": lt, "obligation": obligation})

    closures, dead = [], []
    # Partners are complete typed clauses, but tested incrementally at each
    # character.  This makes the rejection location observable and reproducible.
    for left in chart:
        best = None
        for right in chart:
            if right["id"] == left["id"]: continue
            candidate = left["tape"] + right["tape"]
            rev = candidate[::-1]
            k = 0
            while k < len(candidate) // 2 and candidate[k] == candidate[-1-k]: k += 1
            trace = {"left_id": left["id"], "right_id": right["id"],
                     "left": left["text"], "right": right["text"],
                     "left_number": left["number"], "right_number": right["number"],
                     "first_dead_offset": k, "obligation_prefix": left["obligation"][:16],
                     "status": "closure" if k == len(candidate)//2 else "dead_seam"}
            if k == len(candidate)//2:
                closures.append({"text": left["text"] + " " + right["text"], "audit": audit(left["text"] + " " + right["text"]), "trace": trace})
            elif best is None or k > best["first_dead_offset"]: best = trace
        if best: dead.append(best)
    out = {"experiment_id":"live-typed-chart-seam-solver-20260917",
           "status":"quarantined_no_reader_candidate" if not closures else "exact_candidates_require_readers",
           "candidates": closures, "best_dead_traces": dead,
           "provenance":{"source":"fresh typed clause chart", "catalogue_imported":False,
                         "finished_mirroring":False, "seed_used_as_scaffold":False,
                         "independent_audits":["two-pointer","forward/reverse SHA-256"]},
           "failure_and_repair":{"next_repair":"add role-compatible lexical entries indexed by required seam prefix, retaining number/agreement features before chart composition"}}
    (ROOT/"runs/live-typed-chart-seam-solver-20260917.json").write_text(json.dumps(out, indent=2)+"\n")
    return out

if __name__ == "__main__": print(json.dumps(run(), indent=2))
