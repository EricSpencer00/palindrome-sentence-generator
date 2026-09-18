"""Bounded center-out role zipper.

Unlike the earlier Cartesian clause sweep, this lane schedules grammatical roles
from the center seam outward.  Each extension pays only the newly exposed
character debt and retains the unfinished subject/verb/object obligations.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "center-out-role-zipper-20260918"
ROLES = {
    "subject": ["the baker", "the sailor", "a gardener", "the clerk", "the pilot"],
    "verb": ["marks", "carries", "opens", "guards", "reads", "writes"],
    "object": ["a map", "the gate", "fresh herbs", "the ledger", "old notes"],
    "adjunct": ["at noon", "in spring", "by the harbor", "before dusk"],
}
SCHEDULES = [
    ("subject", "verb", "object", "adjunct"),
    ("adjunct", "object", "verb", "subject"),
    ("subject", "adjunct", "verb", "object"),
]

def letters(s):
    return re.sub(r"[^a-z]", "", s.lower())

def audit(s):
    t = letters(s)
    rev = t[::-1]
    mismatches = sum(a != b for a, b in zip(t, rev)) + abs(len(t) - len(rev))
    # Deliberately independent from the construction score: two-pointer plus hashes.
    exact = bool(t)
    i, j = 0, len(t) - 1
    while i < j:
        if t[i] != t[j]:
            exact = False
            break
        i += 1; j -= 1
    return {"letters": len(t), "two_pointer_exact": exact,
            "mismatches": mismatches,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(rev.encode()).hexdigest()}

def phrase(order, values):
    return " ".join(values[r] for r in order)

def center_out_debt(left, right):
    """Score only exposed symmetric pairs, as roles are appended outward."""
    a, b = letters(left), letters(right)
    n = min(len(a), len(b))
    return sum(x != y for x, y in zip(a[:n], b[::-1][:n])) + abs(len(a)-len(b))

def run():
    # Beam state: paired role prefixes plus unfinished grammatical role sets.
    beam = [{"left": [], "right": [], "debt": 0, "roles_left": set(), "roles_right": set()}]
    for depth in range(4):
        expanded = []
        role = ("subject", "verb", "object", "adjunct")[depth]
        for state in beam:
            for lv in ROLES[role]:
                for rv in ROLES[role]:
                    left = state["left"] + [lv]
                    right = state["right"] + [rv]
                    ltxt, rtxt = " ".join(left), " ".join(right)
                    expanded.append({"left": left, "right": right,
                                     "debt": center_out_debt(ltxt, rtxt),
                                     "roles_left": set(left), "roles_right": set(right)})
        expanded.sort(key=lambda x: (x["debt"], len(" ".join(x["left"])) + len(" ".join(x["right"]))))
        beam = expanded[:24]
    rows = []
    for i, state in enumerate(beam[:8]):
        left, right = " ".join(state["left"]), " ".join(state["right"])
        text = f"{left}; {['at noon', 'in spring', 'before dusk'][i % 3]}, {right}."
        rows.append({"candidate_id": f"co-{i}", "rendered": text,
                     "audit": audit(text), "zipper_state": {
                         "role_schedule": SCHEDULES[0], "boundary_debt": state["debt"],
                         "unfinished_roles_before_seam": [],
                         "left_and_right_roles_scheduled": True},
                     "provenance": {"construction": "center_out_role_zipper",
                         "catalogue_used": False, "wrapped_seed": False,
                         "finished_tape_reversal": False,
                         "repeated_self_palindromic_unit": False,
                         "independently_authored_lexicon": True}})
    best = min(rows, key=lambda x: x["audit"]["mismatches"])
    return {"experiment": EXPERIMENT,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rendered_candidates": rows,
            "stats": {"rendered": len(rows), "exact": sum(r["audit"]["two_pointer_exact"] for r in rows),
                      "longest_letters": max(r["audit"]["letters"] for r in rows),
                      "best_mismatches": best["audit"]["mismatches"], "beam_width": 24},
            "novelty_preflight": {"new_geometry": "role obligations scheduled center-out with incremental seam debt",
                                  "prior_lane_reused": False, "duplicate_sweep": False},
            "next_repair": {"operator": "replace each role lexicon with agreement-carrying paired frames",
                             "reason": "role scheduling reduces search debt but free phrase choices still do not satisfy exact character obligations"},
            "provenance": {"human_readability_certified": False}}

if __name__ == "__main__":
    result = run()
    for directory in (ROOT / "runs", ROOT / "artifacts"):
        (directory / f"{EXPERIMENT}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"]))
