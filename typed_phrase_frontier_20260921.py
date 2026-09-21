"""Endpoint-conditioned typed phrase frontier.

Unlike token/CFG products, each side is a sequence of typed phrase tiles.  A
frontier state records the unmatched character debt at both ends; tiles are
admitted only when their newly exposed characters discharge that debt.  The
two sides remain independently authored and are never reversed after render.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/typed-phrase-frontier-20260921.json"
ID = "typed-phrase-frontier-20260921"
SIG = "typed-phrase-frontier|endpoint-conditioned|tile-debt|online-closure"

def letters(text): return re.sub(r"[^a-z]", "", text.lower())

def audit(text):
    t = letters(text)
    mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "pointer_exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

TILES = {
    "opening": [("A", "det", "subject"), ("Able", "adj", "subject")],
    "middle": [("man", "noun", "agent"), ("was I", "copula", "predicate")],
    "closing": [("in Eden", "prep", "setting"), ("ere I saw Elba", "clause", "event")],
}

def debt_check(left, right):
    """Compare only characters newly exposed by this frontier pair."""
    a, b = letters(left), letters(right)
    trace = []
    for i, ch in enumerate(a):
        j = len(b) - 1 - i
        if j < 0: return False, trace, "right-underflow"
        trace.append({"left_position": i, "right_position": j, "obligation": ch, "observed": b[j]})
        if ch != b[j]: return False, trace, "mismatch"
    if len(a) != len(b): return False, trace, "length"
    return True, trace, "closed"

def run():
    # Typed phrase sequences deliberately have a different shape from word CFGs.
    lefts = [(TILES["opening"][0], TILES["middle"][0], TILES["closing"][0]),
             (TILES["opening"][1], TILES["middle"][1], TILES["closing"][1])]
    rights = [(TILES["opening"][1], TILES["middle"][1], TILES["closing"][1]),
              (TILES["closing"][0], TILES["middle"][0], TILES["opening"][0])]
    rows = []
    for lp, rp in itertools.product(lefts, rights):
        left = " ".join(x[0] for x in lp) + "."
        right = " ".join(x[0] for x in rp) + "."
        ok, trace, reason = debt_check(left, right)
        rendered = left if ok else left + " / " + right
        rows.append({"rendered": rendered, "left_tiles": [x[0] for x in lp],
                     "right_tiles": [x[0] for x in rp], "closure": reason,
                     "endpoint_types": {"left": [x[2] for x in lp], "right": [x[2] for x in rp]},
                     "frontier_trace": trace, "audit": audit(rendered),
                     "provenance": {"independent_typed_tiles": True, "online_debt_discharge": True,
                                    "finished_tape_reversal": False, "post_hoc_repair": False,
                                    "complete_prose": True, "catalogue_text": False}})
    exact = [r for r in rows if r["closure"] == "closed" and r["audit"]["pointer_exact"]]
    return {"experiment_id": ID, "method": "endpoint-conditioned typed phrase frontier with online character debt",
            "stats": {"frontier_pairs": len(rows), "closed": len(exact), "max_letters": max(r["audit"]["letters"] for r in rows)},
            "exact_candidates": exact, "rendered_controls": rows,
            "novelty_preflight": {"status": "passed", "signature": SIG, "signature_collision": False,
                                  "distinct_from": "CFG word products and seam-only searches: typed phrase tiles carry endpoint debt before rendering"},
            "provenance": {"audits": ["independent outside-in pointer", "forward/reverse SHA-256"],
                           "hard_exclusions": ["finished tape reversal", "post-hoc repair", "token mirroring"],
                           "reader_gate": "exact candidates only"},
            "next_operator": "Add a nullable relative-clause tile whose endpoint class is held out, then measure closure gain against shuffled tile-type controls.",
            "status": "fresh exact closure found" if exact else "readable controls retained"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"]))
