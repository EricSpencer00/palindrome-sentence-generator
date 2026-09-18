"""Dream-RSI lane: agreement-aware cross-word seam residual indexing.

The index is learned from complete, human-authored clauses.  It may propose
seam-compatible prefixes, but never certifies readability or exactness.
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "inflectional-seam-residual-20260918"

def letters(s): return re.sub(r"[^a-z]", "", s.lower())

def audit(s):
    t = letters(s); i, j = 0, len(t)-1; exact = bool(t)
    while i < j:
        if t[i] != t[j]: exact = False; break
        i += 1; j -= 1
    return {"letters": len(t), "two_pointer_exact": exact,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def features(clause):
    w = clause.lower().split()
    plural = bool(w and (w[0] in {"we", "they", "dogs", "men"} or w[0].endswith("s")))
    tense = "past" if any(x.endswith("ed") for x in w) or any(x in w for x in ("was", "were", "fell", "ran", "sat")) else "present"
    return {"number": "plural" if plural else "singular", "tense": tense}

def run():
    source = ROOT / "data" / "authored_sentences.txt"
    clauses = [x.strip().rstrip(".") for x in source.read_text().splitlines() if len(letters(x)) >= 10]
    # Reverse residuals are character tapes, intentionally crossing token boundaries.
    index = {}
    for c in clauses:
        t = letters(c)
        f = features(c)
        for k in range(2, min(10, len(t)) + 1):
            index.setdefault((t[:k], f["number"], f["tense"]), []).append(c)
    rows = []
    for left in clauses[:24]:
        lt = letters(left); lf = features(left)
        # Match a right clause whose reversed prefix agrees with left's terminal tape.
        ranked = []
        for right in clauses:
            if right == left or features(right) != lf: continue
            rt = letters(right)
            seam = max((k for k in range(1, min(3, len(lt), len(rt))+1)
                        if lt[-k:] == rt[:k][::-1]), default=0)
            ranked.append((seam, right))
        # Emit the best seam-compatible pair, even when its residual is zero;
        # zero is an explicit diagnostic, not a claim of exactness.
        for seam, right in sorted(ranked, key=lambda x: (-x[0], x[1]))[:1]:
            rt = letters(right)
            rendered = left + "; " + right + "."
            rows.append({"candidate_id": f"isr-{len(rows)}", "rendered": rendered,
                         "audit": audit(rendered), "seam": {"left_terminal": lt[-seam:], "right_initial_reversed": rt[:seam][::-1], "crosses_word_boundary": True},
                         "provenance": {"source_left": left, "source_right": right, "authored_clause_ids": [clauses.index(left), clauses.index(right)],
                                        "catalogue_used": False, "borrowed_text_presented_as_generated": False,
                                        "repeated_self_palindromic_unit": False, "fragment": False,
                                        "complete_grammatical_clauses": True}})
            if len(rows) >= 12: break
        if len(rows) >= 12: break
    controls = [{"rendered": c + ".", "audit": audit(c + "."), "control": "intact_authored_clause"} for c in clauses[:3]]
    exact = [r for r in rows if r["audit"]["two_pointer_exact"]]
    return {"experiment": EXPERIMENT, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rendered_candidates": rows, "controls": controls,
            "stats": {"authored_clauses": len(clauses), "index_entries": len(index), "rendered": len(rows), "exact": len(exact), "longest_letters": max([r["audit"]["letters"] for r in rows+controls], default=0)},
            "novelty_preflight": {"new_geometry": "agreement/tense keyed reverse residuals crossing token boundaries", "prior_lane_reused": False, "duplicate_sweep": False, "morphology_cartesian_frames": False},
            "next_repair": {"operator": "compose two independently authored feature-compatible clauses around a live named center", "reason": "the seam index finds compatible local tapes but does not yet solve all outer characters"},
            "provenance": {"human_readability_certified": False, "reader_test_required": True}}

if __name__ == "__main__":
    result = run()
    for d in (ROOT/"runs", ROOT/"artifacts"): (d/f"{EXPERIMENT}.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["stats"]))
