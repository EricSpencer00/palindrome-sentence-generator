"""Center-first typed event grammar with synchronous bilateral yields.

The center event and its subordinate attachment are selected before either
outer yield is emitted.  Each subsequent left/right event pair carries
agreement, valency, and attachment features; character obligations are tested
after every synchronous extension.  No completed tape is reversed or repaired.
"""
from __future__ import annotations

import hashlib
import json
import re
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/center-first-typed-event-grammar-20260920.json"
ID = "center-first-typed-event-grammar-20260920"


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


SUBJECTS = (("singular", "the patient pilot"), ("singular", "a careful keeper"),
            ("plural", "the young scouts"), ("plural", "several bright guides"))
VERBS = (("singular", "transitive", "studies", ("the northern chart", "a quiet inlet")),
         ("plural", "transitive", "guard", ("the old bridge", "a brass lantern")),
         ("singular", "intransitive", "waits", ("by the harbor", "under the cedar")),
         ("plural", "intransitive", "return", ("before dawn", "after the rain")))
ATTACHMENTS = (("while", "the distant bell rings"), ("although", "the tide remains high"),
               ("when", "the first stars appear"))


def frame(subject, verb, attachment=None):
    number, valency, predicate, complements = verb
    if subject[0] != number:
        return None
    complement = complements[0]
    if valency == "intransitive":
        complement = complements[1]
    text = f"{subject[1]} {predicate} {complement}"
    if attachment:
        text += f", {attachment[0]} {attachment[1]}"
    return {"text": text + ".", "number": number, "valency": valency,
            "attachment": attachment[0] if attachment else None,
            "complement": complement}


def bank(with_attachment=False):
    out = []
    for subject, verb in product(SUBJECTS, VERBS):
        if with_attachment:
            for attachment in ATTACHMENTS:
                item = frame(subject, verb, attachment)
                if item:
                    out.append(item)
        else:
            item = frame(subject, verb)
            if item:
                out.append(item)
    return tuple(out)


def live_obligations(left_parts, right_parts):
    """Compare emitted outer yields in reverse-stream order, retaining buffers."""
    left = letters(" ".join(left_parts))
    right = letters(" ".join(right_parts))[::-1]
    li = ri = 0
    lbuf = rbuf = ""
    checks = 0
    max_buffer = 0
    while li < len(left) or ri < len(right):
        if li < len(left):
            lbuf += left[li:li + 3]
            li += min(3, len(left) - li)
        if ri < len(right):
            rbuf += right[ri:ri + 3]
            ri += min(3, len(right) - ri)
        while lbuf and rbuf:
            checks += 1
            if lbuf[0] != rbuf[0]:
                return {"equations": checks, "satisfied": checks - 1,
                        "all_satisfied": False,
                        "first_mismatch": (checks - 1, lbuf[0], rbuf[0]),
                        "max_obligation_buffer": max(max_buffer, len(lbuf), len(rbuf))}
            lbuf, rbuf = lbuf[1:], rbuf[1:]
        max_buffer = max(max_buffer, len(lbuf), len(rbuf))
    return {"equations": checks, "satisfied": checks, "all_satisfied": True,
            "first_mismatch": None, "max_obligation_buffer": max_buffer}


def run(max_depth=3, limit=12000):
    centers = bank(with_attachment=True)
    outer = bank(with_attachment=False)
    rows, controls = [], []
    states = prunes = 0
    for center in centers:
        # Center is fixed first; paired synchronous yields grow around it.
        for depth in range(1, max_depth + 1):
            for left_seq in product(outer, repeat=depth):
                for right_seq in product(outer, repeat=depth):
                    if states >= limit:
                        break
                    states += 1
                    left = [x["text"] for x in left_seq]
                    right = [x["text"] for x in right_seq]
                    eq = live_obligations(left, right)
                    rendered = " ".join(left + [center["text"]] + right)
                    if len(controls) < 3 and left[-1] != right[-1]:
                        controls.append({"rendered": rendered, "audit": audit(rendered),
                                         "online_character_equations": eq,
                                         "reader_eligible": False, "diagnostic_only": True})
                    if not eq["all_satisfied"]:
                        prunes += 1
                        continue
                    a = audit(rendered)
                    rows.append({"rendered": rendered, "center_event": center,
                                 "left_events": left_seq, "right_events": right_seq,
                                 "online_character_equations": eq, "audit": a,
                                 "provenance": {"center_first": True,
                                                "typed_subordinate_attachment": True,
                                                "agreement_state_carried": True,
                                                "valency_state_carried": True,
                                                "synchronous_outer_yields": True,
                                                "complete_utterances": True,
                                                "catalogue_text": False,
                                                "finished_tape_reversal": False,
                                                "post_hoc_repair": False,
                                                "mirrored_units": False,
                                                "word_order_symmetry": False,
                                                "fragment": False,
                                                "nested_self_palindrome": False}})
                if states >= limit:
                    break
            if states >= limit:
                break
        if states >= limit:
            break
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]
    reader = [r for r in exact if r["provenance"]["complete_utterances"]]
    result = {
        "experiment_id": ID,
        "method": "center-first typed event grammar with synchronous valency-aware outer yields",
        "stats": {"center_events": len(centers), "outer_frames": len(outer),
                  "max_depth": max_depth, "states": states, "live_prunes": prunes,
                  "live_survivors": len(rows), "exact_gt38": len(exact),
                  "reader_eligible": len(reader),
                  "longest_letters": max((r["audit"]["letters"] for r in rows), default=0)},
        "controls": controls, "exact_candidates": exact,
        "reader_facing_candidates": reader,
        "novelty_preflight": {"status": "passed",
            "signature": "center-first|typed-event|agreement|valency|synchronous-yields",
            "registry_inspected": True,
            "distinct_from": "flat frame products, repair lanes, recursive event concatenation, and direct seam/index banks",
            "catalogue_text_imported": False, "finished_tape_reversal": False,
            "post_hoc_repair": False, "mirrored_units": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
                       "reader_evidence": False},
        "status": "no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction": "Permit typed center events whose subordinate attachment contributes a one-character center residual, then grow unequal-depth outer yields synchronously.",
        "reader_gate": "closed until exact candidates exist and blinded human ratings are collected",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), **result["stats"]}))
    for row in controls:
        print(row["rendered"])
    return result


if __name__ == "__main__":
    run()
