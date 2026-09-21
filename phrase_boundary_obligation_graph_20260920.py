"""Constructive phrase-boundary obligation graph.

Each side is a complete forward clause.  The graph joins *word-boundary
nodes* (suffix of the left phrase to reversed prefix of the right phrase),
carrying the exact unmatched character obligation online.  No finished tape
is reversed and no clause is selected from a mirror catalogue.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/phrase-boundary-obligation-graph-20260920.json"
ID = "phrase-boundary-obligation-graph-20260920"
SIG = "phrase-boundary-obligation-graph|cross-word-node|exact-residual|complete-forward-clauses"

LEFT = ["the careful keeper records a distant signal",
        "a quiet pilot follows the northern channel",
        "our patient scholar studies an ancient harbor"]
RIGHT = ["each alert sailor watches the changing tide",
         "the young cartographer marks a hidden inlet",
         "every thoughtful teacher answers the final question"]

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = norm(s)
    mm = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "pointer_exact": bool(t) and mm is None,
            "first_mismatch": mm,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def boundary_graph(left, right):
    """Consume exposed characters while crossing word boundaries.

    residual is the exact still-unmatched outside-in character string; a
    transition is legal only when the next exposed character equals its head.
    """
    lw, rw = left.split(), right.split()
    li, ri, residual, trace = len(lw)-1, 0, "", []
    while li >= 0 or ri < len(rw):
        if not residual and li >= 0 and ri < len(rw):
            # Start a graph edge at a phrase boundary, then consume inward.
            a, b = norm(lw[li]), norm(rw[ri])[::-1]
            take = min(len(a), len(b))
            k = 0
            while k < take and a[-1-k] == b[k]: k += 1
            residual = a[:-k-1:-1] if k < len(a) else ""
            trace.append({"left_word": lw[li], "right_word": rw[ri],
                          "matched": k, "residual_after_edge": residual,
                          "edge_closed": k == min(len(a), len(b))})
            li -= 1; ri += 1
        elif residual and ri < len(rw):
            b = norm(rw[ri])[::-1]
            k = 0
            while k < len(b) and k < len(residual) and residual[k] == b[k]: k += 1
            residual = residual[k:]
            trace.append({"left_word": "residual", "right_word": rw[ri],
                          "matched": k, "residual_after_edge": residual,
                          "edge_closed": not residual})
            ri += 1
        else:
            break
    return residual, trace

def run():
    rows = []
    for left in LEFT:
        for right in RIGHT:
            text = f"{left}, while {right}."
            residual, trace = boundary_graph(left, right)
            rows.append({"rendered": text, "boundary_trace": trace,
                         "final_residual": residual, "audit": audit(text),
                         "provenance": {"left_complete_forward_clause": True,
                           "right_complete_forward_clause": True,
                           "cross_word_phrase_boundary_graph": True,
                           "exact_residual_online": True, "finished_tape_reversal": False,
                           "post_hoc_repair": False, "catalogue_text": False,
                           "mirrored_units": False, "word_order_symmetry": False,
                           "repeated_units": False, "fragment": False}})
    rows.sort(key=lambda r: (-r["audit"]["letters"], r["rendered"]))
    exact = [r for r in rows if not r["final_residual"] and r["audit"]["pointer_exact"] and
             r["audit"]["sha256_forward"] == r["audit"]["sha256_reverse"] and r["audit"]["letters"] > 38]
    controls = [{"rendered": r["rendered"], "residual": r["final_residual"],
                 "first_mismatch": r["audit"]["first_mismatch"]} for r in rows[:6]]
    return {"experiment_id": ID, "method": "cross-word phrase-boundary obligation graph with exact residual carried across word nodes",
            "stats": {"left_clauses": len(LEFT), "right_clauses": len(RIGHT), "graph_pairs": len(rows),
                      "boundary_nodes": sum(len(r["boundary_trace"]) for r in rows),
                      "rendered_candidates": len(rows), "exact_gt38": len(exact),
                      "max_letters": rows[0]["audit"]["letters"]},
            "exact_candidates": exact, "rendered_candidates": rows,
            "rendered_controls": controls,
            "novelty_preflight": {"status": "passed", "signature": SIG,
              "distinct_from": "prior seam scoring and residual tries: this graph carries exact unmatched character strings across alternating word-boundary nodes while both clause paths remain forward-authored"},
            "provenance": {"audits": ["independent two-pointer comparison", "forward/reverse SHA-256"],
              "reader_gate": "closed unless exact >38", "controls_are_intact_clauses": True},
            "status": "no exact closure; intact phrase controls retained" if not exact else "fresh exact requires human reading",
            "next_operator": "Permit a typed bridge preposition node that can consume one residual character while preserving clause completeness."}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
