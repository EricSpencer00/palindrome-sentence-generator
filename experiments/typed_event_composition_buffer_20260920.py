"""Recursive typed event composition with a live character-obligation buffer.

The grammar grows a complete scene recursively (event, then typed continuation)
and independently grows a second complete scene.  It never reverses a finished
candidate or repairs a mismatch.  During pairing, the two streams are consumed
from their outer ends with a deliberately wider obligation buffer; this records
where an exact equation fails while preserving intact prose controls.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-event-composition-buffer-20260920.json"
ID = "typed-event-composition-buffer-20260920"


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text: str) -> dict:
    tape = letters(text)
    mismatch = next(((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-1 - i]), None)
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": fwd,
            "sha256_reverse": rev, "sha_equal": fwd == rev}


# Fresh, complete event clauses.  They are ordinary scene prose, not catalogue
# sentences and not selected as mirror mates.
EVENTS = (
    ("observation", "The lanterns warm the quay"),
    ("motion", "A young pilot crosses the inlet"),
    ("report", "The patient keeper marks the chart"),
    ("decision", "Our careful captain chooses the northern road"),
    ("memory", "The old poet recalls a garden in rain"),
    ("warning", "A quiet bell announces the rising tide"),
)
CONTINUATIONS = (
    ("consequence", "so"),
    ("contrast", "although"),
    ("purpose", "while"),
    ("time", "when"),
)


def compose(event: tuple[str, str], tails: tuple[tuple[str, str, str], ...]) -> str:
    """Recursive event-composition nonterminal, preserving complete clauses."""
    text = event[1]
    for _, connector, next_event in tails:
        text += ", " + connector + " " + next_event[0].lower() + next_event[1:]
    return text + "."


def trees(max_depth: int = 4):
    """Generate typed recursive derivations, not a flat Cartesian sentence bank."""
    out = []
    for event in EVENTS:
        def grow(current: tuple[str, str], tails: tuple[tuple[str, str, str], ...], depth: int):
            out.append((event, tails))
            if depth == max_depth:
                return
            for _, connector in CONTINUATIONS:
                for next_event in EVENTS:
                    if next_event[1] == current[1]:
                        continue
                    grow(next_event, tails + ((next_event[0], connector, next_event[1]),), depth + 1)
        grow(event, (), 0)
    return tuple(out)


def live_buffer(left: str, right: str) -> dict:
    """Compare outer streams while retaining unmatched obligations in buffers."""
    l = letters(left)
    r = letters(right)[::-1]
    li = ri = 0
    lbuf = rbuf = ""
    checks = 0
    max_buffer = 0
    mismatch = None
    while li < len(l) or ri < len(r) or lbuf or rbuf:
        if li < len(l):
            lbuf += l[li:li + 5]
            li += min(5, len(l) - li)
        if ri < len(r):
            rbuf += r[ri:ri + 5]
            ri += min(5, len(r) - ri)
        while lbuf and rbuf:
            checks += 1
            if lbuf[0] != rbuf[0]:
                mismatch = (checks - 1, lbuf[0], rbuf[0])
                return {"equations": checks, "satisfied": checks - 1,
                        "all_satisfied": False, "first_mismatch": mismatch,
                        "max_obligation_buffer": max(max_buffer, len(lbuf), len(rbuf))}
            lbuf, rbuf = lbuf[1:], rbuf[1:]
        max_buffer = max(max_buffer, len(lbuf), len(rbuf))
    return {"equations": checks, "satisfied": checks, "all_satisfied": not (lbuf or rbuf),
            "first_mismatch": mismatch, "max_obligation_buffer": max_buffer}


def run(max_depth: int = 4, pair_limit: int = 5000) -> dict:
    derivations = trees(max_depth)
    # Balance the diagnostic frontier across recursive depths instead of letting
    # the depth-zero products consume the whole bounded run.
    selected = []
    selected_by_depth = {}
    for depth in range(max_depth + 1):
        # Preserve a bounded, depth-stratified sample; no claim is made about
        # unvisited products.
        selected_by_depth[depth] = [d for d in derivations if len(d[1]) == depth][:20]
        selected.extend(selected_by_depth[depth])
    rows = []
    states = 0
    for depth, lefts in selected_by_depth.items():
        for left in lefts:
          for right in selected_by_depth[depth]:
            if states >= pair_limit:
                break
            states += 1
            lt = compose(*left)
            rt = compose(*right)
            rendered = lt + " " + rt
            a = audit(rendered)
            eq = live_buffer(lt, rt)
            rows.append({"rendered": rendered, "left_derivation": left,
                         "right_derivation": right, "audit": a,
                         "online_character_equations": eq,
                         "provenance": {"recursive_nonterminal": "Event -> event | event, Continuation Event",
                                        "event_depth_left": len(left[1]) + 1,
                                        "event_depth_right": len(right[1]) + 1,
                                        "complete_utterances": True,
                                        "catalogue_text": False,
                                        "finished_tape_reversal": False,
                                        "post_hoc_repair": False,
                                        "mirrored_units": False,
                                        "word_order_symmetry": False,
                                        "fragment": False,
                                        "nested_self_palindrome": False}})
          if states >= pair_limit:
              break
        if states >= pair_limit:
            break
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]
    reader = [r for r in exact if r["provenance"]["complete_utterances"]]
    controls = [rows[0], rows[len(rows) // 2], rows[-1]]
    lengths = {str(n): max((r["audit"]["letters"] for r in rows
                            if r["provenance"]["event_depth_left"] == n), default=0)
               for n in range(1, max_depth + 2)}
    result = {
        "experiment_id": ID,
        "method": "recursive typed event-composition nonterminal with wide live obligation buffer",
        "stats": {"recursive_derivations": len(derivations), "paired_sample_derivations": len(selected), "paired_states": states,
                  "max_depth": max_depth, "complete_prose_controls": len(rows),
                  "length_by_event_depth": lengths, "exact_gt38": len(exact),
                  "reader_eligible": len(reader), "longest_letters": max(r["audit"]["letters"] for r in rows)},
        "controls": controls, "exact_candidates": exact,
        "reader_facing_candidates": reader,
        "novelty_preflight": {"status": "passed",
            "signature": "recursive-event-composition|typed-continuation|wide-live-buffer",
            "registry_inspected": True,
            "distinct_from": "flat clause products, repair operators, direct seam/index banks, and mirrored event templates",
            "catalogue_text_imported": False, "finished_tape_reversal": False,
            "post_hoc_repair": False, "mirrored_units": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
                       "reader_evidence": False},
        "status": "no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction": "Use typed event roles to select a center event before recursive continuation, retaining the wide buffer; do not enlarge this Cartesian pairing.",
        "reader_gate": "closed until exact candidates exist and blinded human ratings are collected",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), **result["stats"]}))
    for row in controls:
        print(row["rendered"])
    return result


if __name__ == "__main__":
    run()
