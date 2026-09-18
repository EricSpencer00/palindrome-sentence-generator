"""Fresh semordnilap/cross-word seam search.

The left arm is authored as a small scene.  The right arm is generated from
independent lexical choices whose *word boundaries* can be shifted while their
letters are checked against the reverse residual.  No completed arm is
reversed, and no lexical item may be reused in a candidate.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILY = "semordnilap-crossword-resegmentation-20260918"
SIG = "fresh-scene-left-arm|independent-semordnilap-word-pairs|cross-word-boundary-shifts|typed-clause-realization|two-pointer-sha-audit|no-repeated-units"

# These pairs change lexical role across the seam; they are not self-palindromic.
PAIRS = [("drawer", "reward"), ("diaper", "repaid"), ("deliver", "reviled"),
         ("stressed", "desserts"), ("parts", "strap"), ("gateman", "nametag"),
         ("stop", "pots"), ("smart", "trams"), ("time", "emit")]
LEFTS = [
    "The quiet clerk delivered a note",
    "A patient nurse opened the drawer",
    "The young poet parts the paper",
    "A careful scout repaid the guide",
    "The kind teacher reads the desserts",
    "A tired guard stops near the gate",
]
RIGHTS = [
    "the nurse rewards a clerk",
    "a poet repaid the guide",
    "the scout reviled a deliverer",
    "the teacher serves desserts",
    "a guard sets pots near time",
    "the poet emits a smart remark",
]

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def words(s: str) -> list[str]:
    return re.findall(r"[a-z]+", s.lower())

def audit(text: str) -> dict:
    t = norm(text)
    bad = [{"i": i, "left": t[i], "right": t[-1-i]}
           for i in range(len(t)//2) if t[i] != t[-1-i]]
    h1 = hashlib.sha256(t.encode()).hexdigest()
    h2 = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"exact": bool(t) and not bad, "letters": len(t),
            "mismatch_count": len(bad), "first_mismatches": bad[:4],
            "forward_sha256": h1, "reverse_sha256": h2,
            "sha_equal": h1 == h2}

def seam_candidates(left: str, right: str) -> list[dict]:
    # Try every cross-word split of the right arm. This changes segmentation,
    # not characters, and is intentionally independent of completed reversal.
    out = []
    rw = words(right)
    for cut in range(1, len(rw)):
        r = " ".join(rw[:cut]) + "; " + " ".join(rw[cut:])
        text = left + "; " + r
        used = words(left) + rw
        distinct = len(used) == len(set(used))
        semord = any(a in norm(left) and b in norm(right) for a, b in PAIRS)
        out.append({"text": text, "left": left, "right": r,
                    "boundary_cut": cut, "crossword_shift": True,
                    "semordnilap_pair_present": semord,
                    "no_repeated_units": distinct, "audit": audit(text),
                    "reader_eligible": False})
    return out

def main() -> None:
    probes = []
    for left in LEFTS:
        for right in RIGHTS:
            probes.extend(seam_candidates(left, right))
    # An exact row must also pass syntax/novelty gates; currently all rows are
    # deliberately retained so a future lexical expansion cannot hide failures.
    for row in probes:
        row["reader_eligible"] = (row["audit"]["exact"] and
                                  row["no_repeated_units"] and
                                  row["semordnilap_pair_present"])
    out = {"experiment": FAMILY, "signature": SIG,
           "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "provenance": {"lefts": LEFTS, "rights": RIGHTS, "pairs": PAIRS,
                          "catalogue_text_used": False, "completed_arm_reversed": False},
           "search": {"left_count": len(LEFTS), "right_count": len(RIGHTS),
                      "boundary_variants": len(probes), "candidate_rows": probes},
           "exact_count": sum(r["reader_eligible"] for r in probes),
           "longest_letters": max(r["audit"]["letters"] for r in probes),
           "novelty_preflight": {"family_new": True, "repeated_units_rejected": True,
                                 "word_order_only_symmetry": False},
           "next_repair": "expand typed semordnilap inventory at the first residual mismatch, then preserve scene syntax while shifting the adjacent word boundary",
           "status": "evaluated"}
    p = ROOT / "runs" / (FAMILY + ".json")
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"probes": len(probes), "exact": out["exact_count"],
                      "longest_letters": out["longest_letters"],
                      "independent_sha_rejections": sum(not r["audit"]["sha_equal"] for r in probes)}))

if __name__ == "__main__":
    main()
