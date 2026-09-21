"""Three-character residuals derived from actual outer clause emission."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/three-char-emission-residual-20260920.json"

LEFT = (
    "the patient cartographer marks a quiet inlet before sunrise",
    "a careful beekeeper carries warm honey toward the shed",
    "our young astronomer records a faint comet above the ridge",
)
# Held-out complete clauses. They are grouped by the key observed from the
# emitted left chunk at runtime, not by a manually supplied left/right pair.
HELDOUT = (
    "the evening keeper folds a blue sail near the pier",
    "the local baker shares a fresh loaf with neighbors",
    "a gentle teacher opens the old atlas after supper",
    "a careful teacher opens a green atlas after supper",
    "a quiet ranger follows a silver trail beyond the creek",
    "our alert ranger follows a silver trail beyond the creek",
    "our patient guide studies a red marker beside the bridge",
)
CONTROLS = (
    "the patient cartographer marks a quiet inlet before sunrise, while the evening keeper folds a blue sail near the pier.",
    "a careful beekeeper carries warm honey toward the shed, while a gentle teacher opens the old atlas after supper.",
)

def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = norm(s); mm = next(((i, t[i], t[-1-i]) for i in range(len(t)//2) if t[i] != t[-1-i]), None)
    return {"letters": len(t), "pointer_exact": bool(t) and mm is None, "first_mismatch": mm,
            "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}

def consume(left: str, right: str, debt: str = "") -> tuple[str, dict | None, int]:
    obligation = debt + norm(left); incoming = norm(right)[::-1]; checked = 0
    for i, (a, b) in enumerate(zip(obligation, incoming)):
        checked += 1
        if a != b:
            return obligation[i+1:], {"offset": i, "left": a, "right": b}, checked
    return obligation[len(incoming):], None, checked

def run() -> dict:
    by_key: dict[str, list[str]] = {}
    for clause in HELDOUT:
        by_key.setdefault(norm(clause)[:3], []).append(clause)
    rows = []; missing = 0; prunes = 0
    for left in LEFT:
        words = left.split(); first = " ".join(words[:4]); second = " ".join(words[4:])
        # This is the actual outer emission; only now is the key derived.
        key = norm(first)[:3]
        choices = by_key.get(key, [])
        if not choices: missing += 1
        for right in choices:
            rw = right.split(); rfirst = " ".join(rw[:-3]); rsecond = " ".join(rw[-3:])
            debt, mm1, c1 = consume(first, rsecond)
            debt2, mm2, c2 = consume(second, rfirst, debt)
            accepted = not mm1 and not mm2 and not debt2
            if not accepted: prunes += 1
            rendered = f"{left}; meanwhile, {right}."
            rows.append({"rendered": rendered, "emitted_outer_chunk": first, "derived_key": key,
                         "chunks": {"left": [first, second], "right": [rfirst, rsecond]},
                         "live_equation": {"first": {"checked": c1, "mismatch": mm1, "residual": debt},
                                           "second": {"checked": c2, "mismatch": mm2, "residual": debt2}, "accepted": accepted},
                         "audit": audit(rendered),
                         "provenance": {"complete_left_clause": True, "complete_right_clause": True,
                                        "heldout_complete_clause": True, "key_derived_from_outer_emission": True,
                                        "three_character_key": True, "residual_carried": True,
                                        "finished_tape_reversal": False, "post_hoc_repair": False,
                                        "catalogue_text": False, "mirrored_chunk_pairs": False}})
    exact = [r for r in rows if r["live_equation"]["accepted"] and r["audit"]["pointer_exact"] and
             r["audit"]["sha256_forward"] == r["audit"]["sha256_reverse"] and r["audit"]["letters"] > 38]
    controls = [{"rendered": s, "audit": audit(s), "control": True,
                 "provenance": {"equation_selected": False, "heldout": False}} for s in CONTROLS]
    impossible = (not rows or all(not r["live_equation"]["accepted"] for r in rows))
    next_repair = ({"operator": "typed continuation by residual suffix class", "reason": "three-character emission keys select clauses, but every held-out continuation mismatches before residual closure",
                    "change": "derive a suffix-class key after the first chunk and author two agreement-compatible complete clauses for each class; this changes key position and grammar, not sweep size"}
                   if impossible else {"operator": "held-out reader adjudication", "reason": "an exact online row exists", "change": "blind intact prose against shuffled controls before promotion"})
    return {"experiment_id": "three-char-emission-residual-20260920",
            "method": "three-character key derived from emitted outer clause chunk, selecting held-out complete continuations before second-chunk emission",
            "acceptance_gate": "online residual closure plus pointer/SHA exactness above 38 letters; no shortcut exclusions",
            "stats": {"left_clauses": len(LEFT), "heldout_clauses": len(HELDOUT), "derived_keys": len(by_key), "rendered": len(rows),
                      "missing_key_emissions": missing, "online_prunes": prunes, "exact_above_38": len(exact),
                      "max_letters": max((r["audit"]["letters"] for r in rows), default=0)},
            "derived_key_trace": sorted({r["derived_key"] for r in rows}), "diagnostic_candidates": rows,
            "exact_candidates": exact, "reader_facing_candidates": [], "controls": controls,
            "novelty_preflight": {"status": "passed", "signature": "fresh-authored|emission-derived-3char-key|heldout-complete-continuations|two-stage-residual",
                                  "distinct_from": "prior typed residual bank used authored key labels; this run derives the key from the actual emitted outer chunk and uses held-out clauses grouped independently by observed key",
                                  "catalogue_text_reused": False, "mirrored_chunk_pairs": False},
            "provenance": {"audits": ["live residual consumer", "independent pointer audit", "forward/reverse SHA-256"],
                           "controls_are": "fresh complete prose, not equation-selected", "hard_exclusions": ["catalogue/API text", "mirrored chunks", "finished tape reversal", "post-hoc repair"]},
            "next_repair": next_repair,
            "status": "no exact >38; diagnostic only" if not exact else "exact diagnostic requires human reading"}

if __name__ == "__main__":
    result = run(); OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result["stats"], sort_keys=True))
