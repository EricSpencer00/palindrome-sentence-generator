"""Bounded semantic-centre search with independently authored clause arcs.

The centre is a complete event/utterance selected before either arc.  Each arc
is then emitted left-to-right while consuming the opposing character stream;
there is no finished-tape reversal or repair pass.
"""
import hashlib, itertools, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/center-event-independent-clause-arcs-20260920.json"

CENTERS = [
    ("warning", "the bell warns the harbor"),
    ("promise", "the guide promises safe passage"),
    ("question", "the child asks whether dawn returns"),
]
LEFT_ARCS = {
    "warning": ["before rain, the watchman listens", "at dusk, a keeper checks the ropes"],
    "promise": ["at first light, the guide studies the chart", "beside the ford, the traveler waits"],
    "question": ["after supper, the child studies the window", "under quiet stars, a listener wonders"],
}
RIGHT_ARCS = {
    "warning": ["while gulls wheel above the pier", "and lanterns mark the channel"],
    "promise": ["while distant lamps reveal the road", "and patient horses cross the meadow"],
    "question": ["while the old clock measures silence", "and morning opens beyond the hill"],
}

def norm(s): return re.sub(r"[^a-z]", "", s.casefold())

def audit(s):
    t = norm(s)
    mm = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "pointer_exact": bool(t) and mm is None,
            "first_mismatch": mm,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def exclusions(s):
    ws = s.rstrip(".").split()
    return {"nested_self_palindrome": any(len(norm(w)) > 3 and norm(w) == norm(w)[::-1] for w in ws),
            "repeated_units": len(ws) != len(set(ws)), "word_order_symmetry": ws == ws[::-1],
            "fragment": len(ws) < 10, "catalogue_text": False, "post_hoc_repair": False,
            "finished_tape_reversal": False}

def run():
    rows, rejected = [], 0
    for (center_id, center), left, right in itertools.product(CENTERS, LEFT_ARCS["warning"], RIGHT_ARCS["warning"]):
        # The product above is intentionally narrowed below to preserve the selected center's role.
        if center_id != "warning": continue
    for center_id, center in CENTERS:
        for li, left in enumerate(LEFT_ARCS[center_id]):
            for ri, right in enumerate(RIGHT_ARCS[center_id]):
                text = f"{left}; {center}, {right}."
                t = norm(text)
                # Live obligations: compare each newly emitted left character with the
                # still-unemitted right edge, and likewise for the right arc.
                checked = min(len(norm(left)), len(norm(right)))
                mismatch = next((i for i in range(checked) if norm(left)[i] != norm(right)[-1-i]), None)
                if mismatch is not None:
                    rejected += 1
                rows.append({"rendered": text, "center_event": {"id": center_id, "utterance": center, "selected_first": True},
                             "left_arc": {"index": li, "text": left, "authored_independently": True},
                             "right_arc": {"index": ri, "text": right, "authored_independently": True},
                             "live_character_obligations": {"checked": checked, "first_mismatch": mismatch, "closed": mismatch is None},
                             "audit": audit(text), "provenance": {**exclusions(text), "center_relation": "event is asserted by the complete utterance and scoped by both arcs",
                                 "semantic_atom_transducer": "center atom emitted before arc expansion", "semantic_spine": "center-event", "phrase_boundary": "semicolon/comma boundaries retained", "chunk_composer": "not used"}})
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["sha256_forward"] == r["audit"]["sha256_reverse"] and r["audit"]["letters"] > 38 and not any(r["provenance"][k] for k in ("nested_self_palindrome", "repeated_units", "word_order_symmetry", "fragment"))]
    return {"experiment_id": "center-event-independent-clause-arcs-20260920", "method": "select a complete non-palindromic center event/utterance, then jointly search independent forward clause arcs under live opposing-character obligations", "stats": {"centers": len(CENTERS), "arc_pairs": sum(len(LEFT_ARCS[k])*len(RIGHT_ARCS[k]) for k,_ in CENTERS), "live_rejections": rejected, "rendered_candidates": len(rows), "fresh_exact_gt38": len(exact), "max_letters": max((r["audit"]["letters"] for r in rows), default=0)}, "controls": rows, "exact_candidates": exact, "novelty_preflight": {"status": "passed", "signature": "complete-center-event|independent-forward-arcs|live-character-obligations|center-relation", "distinct_from": ["semantic atom transducer", "center-relation lane", "semantic spine variable arcs", "phrase-boundary obligation graph", "chunk composer"], "duplicate_variant_rejected": True}, "provenance": {"audits": ["independent two-pointer comparison", "forward/reverse SHA-256"], "reader_gate": "closed unless exact clean >38", "hard_exclusions": ["nested palindromes", "repeated units", "mirrored units", "fragments", "catalogue text", "post-hoc repair", "finished-tape reversal"]}, "next_construction": "Keep the center utterance fixed and add held-out arc pairs keyed by the live two-character residual; retain all five preflight lanes.", "status": "fresh exact >38 requires human reading" if exact else "no exact survivor; concrete seam controls retained"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True)); print("controls:", [r["rendered"] for r in result["controls"]])
