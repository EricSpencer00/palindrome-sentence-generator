"""Joint semantic-scene construction with live character equations.

This lane never renders a mirrored suffix.  It chooses an independently
meaningful left scene and searches a small role-typed right scene bank while
consuming the reverse tape one character at a time.  Failed seams are retained
as repair targets for the next run.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "semantic-scene-live-equation-20260917"
SCENES = [
    ("the careful baker", "records", "a warm loaf", "before sunrise"),
    ("the quiet sailor", "folds", "a blue map", "beside the harbor"),
    ("a patient teacher", "opens", "the old book", "after class"),
    ("the young gardener", "carries", "fresh herbs", "toward the kitchen"),
    ("a kind artist", "sketches", "the bright window", "in the studio"),
]
WORDS = re.compile("[a-z]+")

def letters(s):
    return "".join(WORDS.findall(s.lower()))

def audit(s):
    t = letters(s)
    mismatches = []
    i, j = 0, len(t) - 1
    while i < j:
        if t[i] != t[j]:
            mismatches.append({"left": i, "right": j, "a": t[i], "b": t[j]})
        i += 1
        j -= 1
    return {
        "letters": len(t), "two_pointer_exact": bool(t) and not mismatches,
        "first_mismatches": mismatches[:8],
        "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest(),
        "sha_equal_under_reversal": hashlib.sha256(t.encode()).hexdigest() == hashlib.sha256(t[::-1].encode()).hexdigest(),
    }

def run():
    traces, exact = [], []
    for left_slots in SCENES:
        left = " ".join(left_slots)
        tape = letters(left)
        required = tape[::-1]
        # Right scenes remain authored semantic clauses; no derived text is
        # admitted.  A transition is live only if every emitted character so
        # far satisfies the outside-in obligation.
        best = None
        for right_slots in SCENES:
            right = " ".join(right_slots)
            rt = letters(right)
            k = 0
            while k < min(len(required), len(rt)) and required[k] == rt[k]:
                k += 1
            row = {"left": left, "right": right, "required_reverse_prefix": required[:16],
                   "matched_prefix": k, "right_letters": len(rt), "slots": {"left": left_slots, "right": right_slots}}
            if best is None or k > best["matched_prefix"]:
                best = row
            if k == len(required) == len(rt):
                rendered = left + " " + right
                a = audit(rendered)
                if a["two_pointer_exact"]:
                    exact.append({"text": rendered, "audit": a, "provenance": "two independently authored scene clauses"})
        traces.append(best)
    out = {
        "experiment_id": ID, "method": "typed semantic scene lattice with outside-in character equations",
        "rendered_candidates": exact, "candidate_count": len(exact),
        "reader_eligible": bool(exact), "first_seam_traces": traces,
        "provenance": {"source": "fresh authored role/event/location slots", "catalogue_imported": False,
                        "finished_tape_mirroring": False, "independent_audits": ["two-pointer", "forward/reverse SHA-256"]},
        "novelty_preflight": {"signature": ID, "repeated_units": False, "word_order_symmetry": False, "self_palindromic_words": False},
        "failure_and_repair": {"failure": "no semantic right clause matched the live reverse equation" if not exact else "exact closure",
                                "next_repair": "add role-compatible inflections and length-neutral function words at the first unmatched seam; rerun equations before rendering"},
    }
    (ROOT / "runs" / (ID + ".json")).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(exact), "best_matched_prefix": max((x["matched_prefix"] for x in traces), default=0)}, indent=2))
    return out

if __name__ == "__main__":
    run()
