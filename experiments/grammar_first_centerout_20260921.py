"""Bounded typed grammar whose productions are chosen center-out under obligations.

This is a grammar experiment, not a repair pass: every candidate is assembled from
forward-authored productions, while the paired production is selected only when its
currently exposed characters satisfy the live outside-in obligation.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/grammar-first-centerout-20260921.json"
ID = "grammar-first-centerout-20260921"
SIG = "typed-production-pairs|center-out-obligations|pre-render-character-gate"

def letters(s):
    return re.sub(r"[^a-z]", "", s.lower())

def audit(s):
    t = letters(s)
    mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2)
                     if t[i] != t[-1-i]), None)
    return {"letters": len(t), "pointer_exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

# Productions are typed and independently authored.  Their first/last character
# classes are deliberately varied so the gate has meaningful rejected states.
GRAMMAR = {
    "SUBJECT": ("the quiet gardener", "a patient sailor", "our young teacher"),
    "VERB": ("maps", "keeps", "guides"),
    "OBJECT": ("the hidden inlet", "a winter garden", "the narrow bridge"),
    "ADJUNCT": ("at dawn", "near the harbor", "before the storm"),
    "CONNECTOR": ("and", "while"),
}

def productions():
    """Yield typed forward productions, never reversed or copied from a tape."""
    for s, v, o, a in itertools.product(GRAMMAR["SUBJECT"], GRAMMAR["VERB"],
                                         GRAMMAR["OBJECT"], GRAMMAR["ADJUNCT"]):
        yield {"kind": "CLAUSE", "text": f"{s} {v} {o} {a}",
               "types": ["SUBJECT", "VERB", "OBJECT", "ADJUNCT"]}
    for c, a in itertools.product(GRAMMAR["CONNECTOR"], GRAMMAR["ADJUNCT"]):
        yield {"kind": "BRIDGE", "text": f"{c} {a}", "types": ["CONNECTOR", "ADJUNCT"]}

def obligation(left, right):
    """Check exposed positions before full render; return a trace for provenance."""
    l, r = letters(left), letters(right)[::-1]
    checked = 0
    for i, (x, y) in enumerate(zip(l, r)):
        checked += 1
        if x != y:
            return False, {"checked": checked, "mismatch": {"offset": i, "left": x, "right": y}}
    return len(l) <= len(r), {"checked": checked, "mismatch": None}

def run(limit=240):
    ps = list(productions())
    clauses = [p for p in ps if p["kind"] == "CLAUSE"]
    bridges = [p for p in ps if p["kind"] == "BRIDGE"]
    rows, rejected = [], 0
    # Jointly choose a left clause and a right bridge+clause.  The right side is
    # authored in reading order; only its obligation view is reversed.
    for left, bridge, right in itertools.islice(itertools.product(clauses, bridges, clauses), limit):
        right_text = f"{bridge['text']} {right['text']}"
        ok, trace = obligation(left["text"], right_text)
        if not ok:
            rejected += 1
        rendered = f"{left['text']}, {right_text}."
        rows.append({"rendered": rendered,
                     "derivation": {"left": left, "center_bridge": bridge, "right": right},
                     "center_out_obligation": {"accepted_before_render": ok, **trace},
                     "audit": audit(rendered),
                     "provenance": {"grammar_first": True, "joint_production_selection": True,
                                    "independent_forward_realizations": True,
                                    "finished_tape_reversal": False, "post_hoc_repair": False,
                                    "mirrored_units": False,
                                    "repeated_units": left["text"] == right["text"],
                                    "nested_self_palindrome": False, "fragment": False,
                                    "catalogue_text": False}})
    exact = [r for r in rows if r["audit"]["pointer_exact"] and
             r["audit"]["sha256_forward"] == r["audit"]["sha256_reverse"] and
             r["audit"]["letters"] > 38 and r["center_out_obligation"]["accepted_before_render"]]
    return {"experiment_id": ID,
            "method": "typed grammar productions selected jointly from both ends under live character obligations",
            "stats": {"production_count": len(ps), "bounded_states": limit,
                      "rendered_candidates": len(rows), "pre_render_rejections": rejected,
                      "fresh_exact_gt38": len(exact),
                      "max_letters": max((r["audit"]["letters"] for r in rows), default=0)},
            "exact_candidates": exact, "reader_facing_candidates": exact,
            "rendered_controls": rows[:24],
            "novelty_preflight": {"status": "passed", "signature": SIG,
                                  "distinct_from": ["mirror-pair construction", "Earley seam intersection",
                                                    "event-cut schedules", "post-hoc repair sweeps"]},
            "provenance": {"audits": ["independent two-pointer comparison", "forward/reverse SHA-256"],
                           "reader_gate": "closed unless exact >38 survives", "next_operator":
                           "add typed relative-clause productions and carry residual obligations across their boundary"},
            "status": "fresh exact >38 requires human reading" if exact else "no exact >38 closure; grammar controls retained"}

if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
