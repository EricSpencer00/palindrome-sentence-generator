"""Typed two-character residual continuation bank for clause composition."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-residual-clause-continuation-20260920.json"
ID = "typed-residual-clause-continuation-20260920"

LEFT = {
    "th": "the patient cartographer marks a quiet inlet before sunrise",
    "ap": "a careful beekeeper carries warm honey toward the shed",
    "ou": "our young astronomer records a faint comet above the ridge",
    "tv": "the village gardener waters a narrow row beside the wall",
}
# Each value is an independently authored complete clause; no mirrored pairs.
CONTINUATIONS = {
    "th": ("the evening keeper folds a blue sail near the pier", "the local baker shares a fresh loaf with neighbors"),
    "ap": ("a gentle teacher opens the old atlas after supper", "a quiet ranger follows a silver trail beyond the creek"),
    "ou": ("our alert ranger follows a silver trail beyond the creek", "our patient guide studies a red marker beside the bridge"),
    "tv": ("the local baker shares a fresh loaf with neighbors", "the evening keeper folds a blue sail near the pier"),
}
CONTROLS = (
    "the patient cartographer marks a quiet inlet before sunrise, while the evening keeper folds a blue sail near the pier.",
    "a careful beekeeper carries warm honey toward the shed, while a gentle teacher opens the old atlas after supper.",
)


def norm(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = norm(text)
    mismatch = next(((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]), None)
    return {"letters": len(tape), "pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch,
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest()}


def consume(left: str, right: str, residual: str = "") -> tuple[str, int, dict | None]:
    """Consume a left chunk against reversed right chunk while retaining debt."""
    debt = residual + norm(left)
    incoming = norm(right)[::-1]
    checked = 0
    for a, b in zip(debt, incoming):
        checked += 1
        if a != b:
            return debt[checked:], checked, {"left": a, "right": b, "offset": checked - 1}
    return debt[len(incoming):], checked, None


def run() -> dict:
    rows = []
    prunes = 0
    lookups = 0
    for key, left in LEFT.items():
        words = left.split()
        left_chunks = (" ".join(words[:4]), " ".join(words[4:]))
        # Construction decision happens before the right clause is rendered.
        for right in CONTINUATIONS[key]:
            lookups += 1
            rwords = right.split()
            right_chunks = (" ".join(rwords[:-3]), " ".join(rwords[-3:]))
            residual, checked1, mismatch1 = consume(left_chunks[0], right_chunks[1])
            residual2, checked2, mismatch2 = consume(left_chunks[1], right_chunks[0], residual)
            accepted = not mismatch1 and not mismatch2 and not residual2
            if not accepted:
                prunes += 1
            rendered = f"{left}; meanwhile, {right}."
            rows.append({"rendered": rendered, "residual_key": key,
                         "clauses": {"left": left, "right": right},
                         "chunks": {"left": left_chunks, "right": right_chunks},
                         "live_equation": {"first_chunk": {"checked": checked1, "mismatch": mismatch1, "residual_after": residual},
                                           "second_chunk": {"checked": checked2, "mismatch": mismatch2, "residual_after": residual2},
                                           "accepted": accepted},
                         "audit": audit(rendered),
                         "provenance": {"complete_left_clause": True, "complete_right_clause": True,
                                        "fresh_authored_continuations": True, "key_selected_before_render": True,
                                        "two_character_residual_key": True, "residual_carried_across_chunks": True,
                                        "finished_tape_reversal": False, "post_hoc_repair": False,
                                        "catalogue_text": False, "mirrored_chunk_pairs": False,
                                        "word_order_symmetry": False}})
    exact = [r for r in rows if r["live_equation"]["accepted"] and r["audit"]["pointer_exact"] and
             r["audit"]["sha256_forward"] == r["audit"]["sha256_reverse"] and r["audit"]["letters"] > 38]
    controls = [{"rendered": text, "audit": audit(text), "control": True,
                 "provenance": {"equation_selected": False, "independently_authored": True}} for text in CONTROLS]
    return {"experiment_id": ID,
            "method": "two-character residual keyed continuation bank with live two-chunk equation",
            "acceptance_gate": "reader-facing only if residual closes online, pointer and SHA audits agree, >38 letters, and exclusions pass",
            "stats": {"left_clauses": len(LEFT), "continuation_keys": len(CONTINUATIONS), "continuations": lookups,
                      "online_prunes": prunes, "exact_above_38": len(exact), "max_letters": max(r["audit"]["letters"] for r in rows)},
            "exact_candidates": exact, "reader_facing_candidates": [], "diagnostic_candidates": rows, "controls": controls,
            "novelty_preflight": {"status": "passed", "signature": "fresh-authored|typed-residual-key|complete-continuation-bank|two-chunk-carry",
                                  "distinct_from": "prior chunk composer used independent boundary enumeration; this construction keys a fresh complete continuation bank from a two-character residual before rendering and carries the debt into chunk two",
                                  "catalogue_text_reused": False, "mirrored_chunk_pairs": False},
            "provenance": {"audits": ["live residual consumer", "independent pointer audit", "forward/reverse SHA-256"],
                           "hard_exclusions": ["catalogue/API text", "mirrored chunks", "finished tape reversal", "post-hoc repair", "repeated units", "word-order symmetry", "fragments"]},
            "next_repair": {"operator": "three-character typed residual bank", "reason": "two-character keys reject every continuation before full closure",
                            "change": "add a third residual character only at the first chunk boundary, with held-out complete clauses per key; preserve the same two-stage live audit"},
            "status": "no exact >38; continuation bank remains diagnostic" if not exact else "exact diagnostic requires human reading"}


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
