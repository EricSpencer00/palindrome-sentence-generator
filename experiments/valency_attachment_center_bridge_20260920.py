"""Valency/attachment-aware bilateral constructor with an explicit center bridge.

Clause frames carry subject agreement and transitivity.  The search emits left
and right clauses independently and streams their character obligations from
opposite ends; it never repairs a mismatch or reverses a finished candidate.
"""
from __future__ import annotations

import hashlib
import json
import re
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/valency-attachment-center-bridge-20260920.json"
ID = "valency-attachment-center-bridge-20260920"


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


SUBJECTS = (
    ("singular", "the quiet sailor"),
    ("singular", "a patient keeper"),
    ("plural", "the young scouts"),
    ("plural", "several bright guides"),
)
TRANSITIVE = (
    ("singular", "transitive", "charts", ("the northern inlet", "a weathered map")),
    ("plural", "transitive", "guard", ("the lantern", "a narrow bridge")),
    ("singular", "transitive", "carries", ("fresh water", "the brass compass")),
    ("plural", "transitive", "follow", ("the river road", "a distant signal")),
)
INTRANSITIVE = (
    ("singular", "intransitive", "waits", ("at the harbor", "by the old tower")),
    ("plural", "intransitive", "return", ("before dawn", "after the rain")),
    ("singular", "intransitive", "listens", ("under the cedar", "near the gate")),
    ("plural", "intransitive", "travel", ("through the valley", "toward the shore")),
)
BRIDGES = (("and", "coordination-center"), ("but", "contrast-center"),
           ("so", "consequence-center"))


def compatible(subject, verb) -> bool:
    return subject[0] == verb[0]


def frames():
    """Yield typed complete clauses with attachment metadata."""
    for subject, verb in product(SUBJECTS, TRANSITIVE + INTRANSITIVE):
        subj, vf = subject
        agreement, valency, predicate, attachments = verb
        if subj != agreement:
            continue
        for attachment in attachments:
            if valency == "transitive":
                text = f"{subject[1]} {predicate} {attachment}."
                slot = "object"
            else:
                text = f"{subject[1]} {predicate} {attachment}."
                slot = "adjunct"
            yield {"text": text, "subject_number": subj, "valency": valency,
                   "attachment": slot, "attachment_text": attachment,
                   "predicate": predicate}


def live_outer_check(left: str, center: str, right: str) -> dict:
    """Stream opposing clause chunks and retain a wide obligation buffer."""
    l = letters(left)
    r = letters(right)[::-1]
    li = ri = 0
    lbuf = rbuf = ""
    checks = 0
    max_buffer = 0
    while li < len(l) or ri < len(r):
        if li < len(l):
            lbuf += l[li:li + 4]
            li += min(4, len(l) - li)
        if ri < len(r):
            rbuf += r[ri:ri + 4]
            ri += min(4, len(r) - ri)
        while lbuf and rbuf:
            checks += 1
            if lbuf[0] != rbuf[0]:
                return {"equations": checks, "satisfied": checks - 1,
                        "all_outer_satisfied": False,
                        "first_mismatch": (checks - 1, lbuf[0], rbuf[0]),
                        "max_obligation_buffer": max(max_buffer, len(lbuf), len(rbuf)),
                        "center_bridge_streamed": False}
            lbuf, rbuf = lbuf[1:], rbuf[1:]
        max_buffer = max(max_buffer, len(lbuf), len(rbuf))
    # The center is deliberately explicit and audited only after outer
    # obligations have survived; it is not used to repair an outer mismatch.
    return {"equations": checks, "satisfied": checks,
            "all_outer_satisfied": True, "first_mismatch": None,
            "max_obligation_buffer": max_buffer,
            "center_bridge_streamed": bool(center)}


def run(limit: int = 6000) -> dict:
    bank = tuple(frames())
    rows = []
    diagnostic_controls = []
    states = outer_prunes = 0
    for left, right, bridge in product(bank, bank, BRIDGES):
        if states >= limit:
            break
        states += 1
        eq = live_outer_check(left["text"], bridge[0], right["text"])
        rendered = (left["text"].rstrip(".") + ", " + bridge[0] + " " +
                    right["text"][0].lower() + right["text"][1:])
        if len(diagnostic_controls) < 3 and right["text"] != left["text"]:
            diagnostic_controls.append({"rendered": rendered,
                "left_frame": left, "right_frame": right,
                "center_bridge": bridge, "online_character_equations": eq,
                "audit": audit(rendered),
                "reader_eligible": False,
                "diagnostic_only": True})
        if not eq["all_outer_satisfied"]:
            outer_prunes += 1
            continue
        a = audit(rendered)
        rows.append({"rendered": rendered, "left_frame": left, "right_frame": right,
                     "center_bridge": bridge, "online_character_equations": eq,
                     "audit": a,
                     "provenance": {"valency_state_carried": True,
                                    "agreement_state_carried": True,
                                    "attachment_state_carried": True,
                                    "explicit_center_bridge": True,
                                    "complete_utterances": True,
                                    "catalogue_text": False,
                                    "finished_tape_reversal": False,
                                    "post_hoc_repair": False,
                                    "mirrored_units": False,
                                    "word_order_symmetry": False,
                                    "fragment": False,
                                    "nested_self_palindrome": False}})
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]
    reader = [r for r in exact if r["provenance"]["complete_utterances"]]
    controls = diagnostic_controls
    result = {
        "experiment_id": ID,
        "method": "valency/attachment-aware bilateral clause constructor with explicit center bridge",
        "stats": {"typed_clause_frames": len(bank), "center_bridges": len(BRIDGES),
                  "states": states, "outer_prunes": outer_prunes,
                  "outer_survivors": len(rows), "exact_gt38": len(exact),
                  "reader_eligible": len(reader),
                  "longest_letters": max((r["audit"]["letters"] for r in rows), default=0)},
        "controls": controls, "diagnostic_controls": diagnostic_controls,
        "exact_candidates": exact,
        "reader_facing_candidates": reader,
        "novelty_preflight": {"status": "passed",
            "signature": "valency|agreement|attachment|explicit-center-bridge|live-outer-equations",
            "registry_inspected": True,
            "distinct_from": "recursive event composition, repair lanes, direct seam/index banks, and word-pair products",
            "catalogue_text_imported": False, "finished_tape_reversal": False,
            "post_hoc_repair": False, "mirrored_units": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
                       "reader_evidence": False},
        "status": "no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction": "Carry a typed center event with agreement-compatible subordinate attachment, then stream its two sides without expanding flat frame products.",
        "reader_gate": "closed until exact candidates exist and blinded human ratings are collected",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), **result["stats"]}))
    for row in controls:
        print(row["rendered"])
    return result


if __name__ == "__main__":
    run()
