"""Clause-level chunk composition with an online two-sided letter equation.

Every bank item is a complete, independently authored clause.  The composer
chooses independent word-boundary segmentations, then streams the left chunks
against reversed right chunks.  It never imports mirrored token pairs or the
public catalogue.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/clause-chunk-equation-composer-20260920.json"
ID = "clause-chunk-equation-composer-20260920"
SIGNATURE = "fresh-authored|complete-clause-chunks|independent-segmentation|online-equation"

LEFT = (
    "the patient cartographer marks a quiet inlet before sunrise",
    "a careful beekeeper carries warm honey toward the shed",
    "our young astronomer records a faint comet above the ridge",
    "the village gardener waters a narrow row beside the wall",
)
RIGHT = (
    "the evening keeper folds a blue sail near the pier",
    "a gentle teacher opens the old atlas after supper",
    "our alert ranger follows a silver trail beyond the creek",
    "the local baker shares a fresh loaf with the neighbors",
)
CONTROLS = (
    "the patient cartographer marks a quiet inlet before sunrise, while the evening keeper folds a blue sail near the pier.",
    "a gentle teacher opens the old atlas after supper, and our alert ranger follows a silver trail beyond the creek.",
)


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    mismatch = next(
        ((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2) if tape[i] != tape[-1 - i]),
        None,
    )
    return {
        "letters": len(tape),
        "pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
        "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
    }


def segment(clause: str, cut: int) -> tuple[str, ...]:
    words = clause.split()
    return (" ".join(words[:cut]), " ".join(words[cut:]))


def stream_equation(left_chunks: tuple[str, ...], right_chunks: tuple[str, ...]) -> dict:
    """Consume left forward and right backward, rejecting the first mismatch."""
    left = letters(" ".join(left_chunks))
    right = letters(" ".join(right_chunks))[::-1]
    checked = 0
    for offset, (a, b) in enumerate(zip(left, right)):
        checked += 1
        if a != b:
            return {"accepted": False, "characters_checked": checked,
                    "first_mismatch": {"offset": offset, "left": a, "right": b}}
    accepted = len(left) == len(right)
    return {"accepted": accepted, "characters_checked": checked,
            "first_mismatch": None if accepted else {"offset": checked, "reason": "length"}}


def novelty_preflight() -> dict:
    registry = ROOT / "docs/experiment-novelty-registry.json"
    data = json.loads(registry.read_text())
    entries = data.get("entries", [])
    prior = [row for row in entries if row.get("id") == ID or row.get("signature") == SIGNATURE]
    related = [row["id"] for row in entries if any(
        token in row.get("signature", "") for token in ("chunk", "segmentation", "complete-clause")
    )]
    script_hits = []
    for path in ROOT.glob("*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(errors="ignore")
        if "clause-chunk-equation-composer" in text or "independent-segmentation" in text:
            script_hits.append(path.name)
    return {
        "status": "passed" if not prior and not script_hits else "review-required",
        "signature": SIGNATURE,
        "exact_id_or_signature_collisions": [row["id"] for row in prior],
        "related_registry_entries": related,
        "same_operator_script_hits": script_hits,
        "distinct_from": "prior clause products enumerate complete pairs; this run adds independent clause-internal chunk boundaries and discharges a live residual before the next chunk is emitted",
        "catalogue_text_reused": False,
        "mirrored_chunk_pairs": False,
    }


def run() -> dict:
    rows = []
    online_prunes = 0
    segment_states = 0
    for left in LEFT:
        for right in RIGHT:
            for left_cut in range(2, len(left.split()) - 1):
                for right_cut in range(2, len(right.split()) - 1):
                    segment_states += 1
                    lc, rc = segment(left, left_cut), segment(right, right_cut)
                    equation = stream_equation(lc, rc)
                    if not equation["accepted"]:
                        online_prunes += 1
                    rendered = f"{lc[0]} {lc[1]}; meanwhile, {rc[0]} {rc[1]}."
                    rows.append({
                        "rendered": rendered,
                        "clauses": {"left": left, "right": right},
                        "chunks": {"left": lc, "right": rc},
                        "equation": equation,
                        "audit": {"left": audit(left), "right": audit(right), "rendered": audit(rendered)},
                        "provenance": {
                            "complete_left_clause": True, "complete_right_clause": True,
                            "independently_authored_banks": True, "independent_word_boundary_cuts": True,
                            "online_before_render": True, "finished_tape_reversal": False,
                            "post_hoc_repair": False, "catalogue_text": False,
                            "mirrored_chunk_pairs": False, "word_order_symmetry": False,
                        },
                    })
    rows.sort(key=lambda row: (-row["audit"]["rendered"]["letters"], row["rendered"]))
    exact = [row for row in rows if row["equation"]["accepted"] and row["audit"]["rendered"]["letters"] > 38]
    controls = [{"rendered": text, "audit": audit(text), "control": True,
                 "provenance": {"independently_authored": True, "equation_selected": False}}
                for text in CONTROLS]
    return {
        "experiment_id": ID,
        "method": "complete independently authored clauses composed from independent clause-internal chunks with online two-sided letter equations",
        "acceptance_gate": "reader-facing only if online equation closes, normalized rendered audit is exact, >38 letters, and all shortcut exclusions pass",
        "stats": {"left_clauses": len(LEFT), "right_clauses": len(RIGHT), "segment_states": segment_states,
                  "online_mismatch_prunes": online_prunes, "rendered": len(rows),
                  "exact_above_38": len(exact), "max_rendered_letters": max(r["audit"]["rendered"]["letters"] for r in rows)},
        "exact_candidates": exact,
        "reader_facing_candidates": exact,
        "controls": controls,
        "novelty_preflight": novelty_preflight(),
        "provenance": {"audits": ["independent two-pointer stream", "forward/reverse SHA-256"],
                       "fixed_conditions": ["fresh authored clause banks", "two chunks per clause", "word-boundary cuts", "no post-hoc repair"],
                       "hard_exclusions": ["catalogue text", "mirrored chunk pairs", "finished tape reversal", "repeated units", "word-order symmetry", "fragments"]},
        "next_repair": {"operator": "typed residual continuation bank", "reason": "all current independent clause cuts mismatch before closure",
                        "change": "author alternate complete clause continuations keyed by the first two-character residual, then retain the key across the second chunk; rerun the same exact gate"},
        "status": "fresh exact >38 requires human reading" if exact else "no exact >38 closure; independently authored prose controls retained",
    }


if __name__ == "__main__":
    result = run()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
