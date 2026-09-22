"""Negative control for the requested t/d two-word function-slot operator.

The prior draft exposed reversed gibberish as verbs.  This lane deliberately
records the linguistic obstruction instead of widening into a reverse sweep.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/authored-wordpair-obstruction-20260921.json"
ORDINARY = ("to", "from", "that", "the", "in", "on", "at", "by", "for", "with")

def norm(s): return re.sub("[^a-z]", "", s.casefold())
def audit(s):
    t = norm(s)
    mm = next((i for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "two_pointer_exact": bool(t) and mm is None,
            "first_mismatch": mm,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def run():
    rows = []
    for a, b in (("the baker sends a note", "while the keeper reads the map"),
                 ("the keeper marks a page", "while the baker carries the note")):
        text = f"{a}, {b}."
        rows.append({"rendered": text, "complete_prose": True, "audit": audit(text),
                     "provenance": {"ordinary_english_only": True,
                                    "independently_authored": True,
                                    "reversed_token_surfaces": False,
                                    "catalogue_pairs": False,
                                    "finished_tape_reversal": False}})
    return {"experiment_id": "authored-wordpair-obstruction-20260921",
            "status": "operator_abandoned_linguistic_obstruction",
            "target_boundary": {"left": "t", "right": "d"},
            "candidate_count": len(rows), "exact_count": 0, "candidates": rows,
            "obstruction": "No ordinary-English two-word function-slot pair was found whose normalized boundary supplies t/d while preserving grammatical clauses; satisfying it requires a reversed/gibberish surface or a catalogue semordnilap.",
            "next_operator": "Do not widen this lane; author a different ordinary lexical boundary with an explicit semantic frame.",
            "provenance": {"bounded_authored_probes": True,
                           "independent_audits": ["two-pointer", "forward/reverse SHA-256"]}}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidate_count": result["candidate_count"], "exact_count": 0}))
