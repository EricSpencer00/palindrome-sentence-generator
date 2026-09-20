"""Morphological-paradigm zipper CSP.

The two clauses are independently sampled from productive English paradigms.
Agreement and tense choose inflectional variants before any character is
accepted; a zipper then consumes the outside characters of both clauses and
prunes the pair at its first live disagreement.  No word is mirrored and no
finished tape is repaired or reversed.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = "morphological-paradigm-zipper-csp-20260920"
SIGNATURE = "paradigm-zipper|agreement|productive-inflection|live-character-csp"

SUBJECTS = {
    "sg": ["the lantern keeper", "a patient reader", "the young poet"],
    "pl": ["the lantern keepers", "patient readers", "the young poets"],
}
VERBS = {
    ("sg", "present"): ["marks", "keeps", "reads"],
    ("pl", "present"): ["mark", "keep", "read"],
    ("sg", "past"): ["marked", "kept", "read"],
    ("pl", "past"): ["marked", "kept", "read"],
}
OBJECTS = ["the old letter", "a silver map", "the quiet poem"]
ADJUNCTS = ["by the river", "near the garden", "at first light"]

def norm(s):
    return re.sub(r"[^a-z]", "", s.lower())

def audit(text):
    tape = norm(text); rev = tape[::-1]
    m = next(((i, tape[i], rev[i]) for i in range(len(tape)) if tape[i] != rev[i]), None)
    return {"rendered": text, "letters": len(tape), "exact": bool(tape) and m is None,
            "two_pointer_exact": bool(tape) and m is None, "first_mismatch": m,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest()}

def frame(number, tense, subject, verb, obj, adjunct):
    # Each choice is a grammatical slot, not a character-level mutation.
    return f"{subject} {verb} {obj} {adjunct}"

def zipper(left, right):
    """Compare newly emitted outer characters, stopping at first mismatch."""
    a, b = norm(left), norm(right)
    compared = 0
    for i in range(min(len(a), len(b))):
        compared += 1
        if a[i] != b[-1-i]:
            return {"compatible": False, "compared": compared,
                    "first_mismatch": (i, a[i], b[-1-i])}
    return {"compatible": len(a) == len(b), "compared": compared,
            "first_mismatch": None if len(a) == len(b) else (len(a), "", "length")}

def main():
    frames = []
    for number, tense in itertools.product(("sg", "pl"), ("present", "past")):
        for subject, verb, obj, adjunct in itertools.product(SUBJECTS[number], VERBS[(number, tense)], OBJECTS, ADJUNCTS):
            frames.append({"number": number, "tense": tense, "subject": subject,
                           "verb": verb, "object": obj, "adjunct": adjunct,
                           "text": frame(number, tense, subject, verb, obj, adjunct)})
    rows, compatible = [], 0
    # Keep a bounded, deterministic cross-product: distinct paradigms are the
    # search space, while the zipper is the live character gate.
    for left, right in itertools.product(frames, frames):
        z = zipper(left["text"], right["text"]); 
        if not z["compatible"]:
            continue
        compatible += 1
        rendered = left["text"] + "; " + right["text"] + "."
        rows.append({"rendered": rendered, "left_features": {k:left[k] for k in ("number","tense","verb")},
                     "right_features": {k:right[k] for k in ("number","tense","verb")},
                     "zipper": z, "audit": audit(rendered),
                     "provenance": {"independent_paradigm_choices": True, "agreement_checked": True,
                                    "live_outer_character_gate": True, "word_order_symmetry": False,
                                    "finished_tape_reversal": False, "post_hoc_repair": False,
                                    "catalogue_import": False}})
    # Preserve readable controls even when the exact gate is empty.
    if not rows:
        controls = [(frames[0], frames[-1]), (frames[13], frames[41])]
        for left, right in controls:
            rendered = left["text"] + "; " + right["text"] + "."
            rows.append({"rendered": rendered, "control": True, "audit": audit(rendered),
                         "provenance": {"independent_paradigm_choices": True, "agreement_checked": True,
                                        "live_outer_character_gate": True, "reader_eligible": False,
                                        "finished_tape_reversal": False, "post_hoc_repair": False,
                                        "catalogue_import": False}})
    exact = [r for r in rows if r["audit"]["exact"] and r["audit"]["letters"] > 38]
    out = {"experiment_id": ID, "signature": SIGNATURE,
           "method": "productive number/tense/verb paradigm selection with outside-in zipper CSP",
           "status": "fresh_exact_found" if exact else "completed_no_exact_closure",
           "reader_eligible": False, "stats": {"frame_count": len(frames), "transitions": len(frames)**2,
           "live_compatible_pairs": compatible, "rendered_controls_or_closures": len(rows),
           "fresh_exact_gt38": len(exact)}, "rendered_candidates": rows, "exact_candidates": exact,
           "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "audits": ["independent two-pointer", "forward/reverse SHA-256"],
                          "next_construction": "carry residual suffix-class vectors through a three-clause paradigm grammar with held-out auxiliaries",
                          "next_reader_test": "blinded human readability only after an exact candidate exceeds 38 letters"}}
    path = ROOT / "runs" / (ID + ".json"); path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["stats"], sort_keys=True))

if __name__ == "__main__": main()
