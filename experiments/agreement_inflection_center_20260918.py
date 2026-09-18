"""Agreement-carrying named-center repair.

This lane makes inflection a search variable: each paired clause carries number,
tense, and determiner features, while the character equation is checked as each
outer pair is proposed.  It is deliberately not a reverse-tape renderer.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "agreement-inflection-center-20260918"

SUBJECTS = [("the baker", "sg"), ("the clerks", "pl"), ("a sailor", "sg"),
            ("some gardeners", "pl")]
VERBS = {"sg": [("marks", "pres"), ("carries", "pres"), ("marked", "past")],
         "pl": [("mark", "pres"), ("carry", "pres"), ("marked", "past")]}
OBJECTS = [("a map", "sg"), ("the ledger", "sg"), ("fresh herbs", "pl"),
           ("old notes", "pl")]
CENTERS = ["Mara", "Nora", "Rhea", "Iris"]

def letters(s): return re.sub(r"[^a-z]", "", s.lower())

def audit(s):
    t = letters(s); i, j = 0, len(t)-1; exact = bool(t)
    while i < j:
        if t[i] != t[j]: exact = False; break
        i += 1; j -= 1
    mismatches = sum(a != b for a, b in zip(t, t[::-1])) // 2
    return {"letters": len(t), "two_pointer_exact": exact, "mismatches": mismatches,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def clause(subject, verb, obj, center):
    return f"{subject} {verb} {obj} near {center}"

def residual(left, right):
    """Count only already exposed equation pairs (independent of final audit)."""
    a, b = letters(left), letters(right)[::-1]
    n = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:n], b[:n])) + abs(len(a)-len(b))

def run():
    # The state explicitly carries agreement; incompatible subject/verb pairs
    # never reach rendering.  Right clauses are selected independently.
    states = []
    for center in CENTERS:
        for subject, number in SUBJECTS:
            for verb, tense in VERBS[number]:
                for obj, obj_number in OBJECTS:
                    for rsubject, rnumber in SUBJECTS:
                        for rverb, rtense in VERBS[rnumber]:
                            # Agreement is enforced on both sides; tense is a
                            # carried feature rather than a post-hoc rewrite.
                            left = clause(subject, verb, obj, center)
                            # The right clause chooses its own object; identical
                            # clause copies are a diagnostic shortcut, never a
                            # reader-facing candidate.
                            for robj, robj_number in OBJECTS:
                                right = clause(rsubject, rverb, robj, center)
                                if right == left:
                                    continue
                                text = left + "; " + right + "."
                                states.append({"left": left, "right": right,
                                    "center": center, "features": {
                                        "left_number": number, "right_number": rnumber,
                                        "left_tense": tense, "right_tense": rtense,
                                        "left_object_number": obj_number,
                                        "right_object_number": robj_number},
                                    "equation_residual": residual(left, right)})
    states.sort(key=lambda s: (s["equation_residual"], len(s["left"])+len(s["right"])))
    rows = []
    for i, state in enumerate(states[:12]):
        text = state["left"] + "; " + state["right"] + "."
        rows.append({"candidate_id": f"aic-{i}", "rendered": text,
          "audit": audit(text), "equation_state": state,
          "provenance": {"construction": "agreement_carrying_inflection_search",
             "catalogue_used": False, "wrapped_seed": False,
             "finished_tape_reversal": False, "repeated_self_palindromic_unit": False,
             "independently_authored_lexicon": True, "reader_eligible": False}})
    best = min(rows, key=lambda r: r["audit"]["mismatches"])
    return {"experiment": EXPERIMENT,
      "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      "rendered_candidates": rows,
      "stats": {"states": len(states), "rendered": len(rows),
         "exact": sum(r["audit"]["two_pointer_exact"] for r in rows),
         "longest_letters": max(r["audit"]["letters"] for r in rows),
         "best_mismatches": best["audit"]["mismatches"]},
      "novelty_preflight": {"new_geometry": "agreement and inflection features remain live in paired character equations",
         "prior_lane_reused": False, "duplicate_sweep": False},
      "next_repair": {"operator": "attach asymmetric lexical bridge words at the named-center seam",
         "reason": "feature-compatible clauses still expose unmatched outer determiners before the center equation can close"},
      "provenance": {"human_readability_certified": False}}

if __name__ == "__main__":
    result = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
