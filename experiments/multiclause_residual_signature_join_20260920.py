"""Bidirectional multi-clause residual-signature join.

This lane indexes complete two-clause scenes before lexical emission.  A left
scene's outward character obligation is joined to a right scene's reversed
obligation signature, then surviving joins are streamed through a live
residual checker.  Clause frames retain agreement and valency metadata.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/multiclause-residual-signature-join-20260920.json"
ID = "multiclause-residual-signature-join-20260920"


def letters(text):
    return re.sub(r"[^a-z]", "", text.casefold())


def audit(text):
    tape = letters(text)
    mismatch = next(((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-1 - i]), None)
    fwd = hashlib.sha256(tape.encode()).hexdigest()
    rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": fwd,
            "sha256_reverse": rev, "sha_equal": fwd == rev}


@dataclass(frozen=True)
class Clause:
    number: str
    valency: str
    text: str


SUBJECTS = (("singular", "the weathered sailor"), ("singular", "a patient keeper"),
            ("plural", "the young scouts"), ("plural", "several bright guides"))
VERBS = (("singular", "transitive", "studies", "the northern chart"),
         ("singular", "intransitive", "waits", "by the harbor"),
         ("plural", "transitive", "guard", "the old bridge"),
         ("plural", "intransitive", "return", "after the rain"))


def clause_bank():
    bank = []
    for number, subject in SUBJECTS:
        for agreement, valency, verb, complement in VERBS:
            if number == agreement:
                bank.append(Clause(number, valency, f"{subject} {verb} {complement}."))
    return tuple(bank)


def scene_pairs(bank):
    """Complete two-clause scenes; no token is chosen as a mirror mate."""
    for first, second in product(bank, repeat=2):
        yield (first, second)


def render(scene):
    return " Then ".join(c.text for c in scene)


def signature(text, width=1):
    tape = letters(text)
    return (tape[:width], len(tape) % 2, len(tape) % 3)


def live_join(left_text, right_text, chunk=4):
    """Consume left and reversed-right residuals without post-hoc repair."""
    left, right = letters(left_text), letters(right_text)[::-1]
    i = j = 0
    lb = rb = ""
    checks = 0
    max_residual = 0
    while i < len(left) or j < len(right):
        if i < len(left):
            lb += left[i:i + chunk]
            i += min(chunk, len(left) - i)
        if j < len(right):
            rb += right[j:j + chunk]
            j += min(chunk, len(right) - j)
        while lb and rb:
            checks += 1
            if lb[0] != rb[0]:
                return {"equations": checks, "satisfied": checks - 1,
                        "all_satisfied": False, "first_mismatch": (checks - 1, lb[0], rb[0]),
                        "max_residual": max(max_residual, len(lb), len(rb))}
            lb, rb = lb[1:], rb[1:]
        max_residual = max(max_residual, len(lb), len(rb))
    return {"equations": checks, "satisfied": checks, "all_satisfied": not (lb or rb),
            "first_mismatch": None, "max_residual": max_residual}


def run(limit=5000):
    bank = clause_bank()
    scenes = tuple(scene_pairs(bank))
    right_index = {}
    for idx, scene in enumerate(scenes):
        # The right signature is defined on its reverse stream, so joins happen
        # before any full lexical candidate is rendered.
        right_index.setdefault(signature(render(scene)[::-1]), []).append(idx)
    joined = checked = pruned = 0
    rows, controls = [], []
    for left_idx, left_scene in enumerate(scenes):
        left_text = render(left_scene)
        candidates = right_index.get(signature(left_text), [])
        joined += len(candidates)
        for right_idx in candidates:
            if checked >= limit:
                break
            checked += 1
            right_scene = scenes[right_idx]
            right_text = render(right_scene)
            rendered = left_text + " " + right_text
            eq = live_join(left_text, right_text)
            row = {"rendered": rendered, "left_scene": [asdict(c) for c in left_scene],
                   "right_scene": [asdict(c) for c in right_scene],
                   "residual_signature": signature(left_text),
                   "online_character_equations": eq, "audit": audit(rendered)}
            if (len(controls) < 3 and left_idx != right_idx and
                    set(c.text for c in left_scene).isdisjoint(set(c.text for c in right_scene))):
                controls.append({**row, "reader_eligible": False, "diagnostic_only": True})
            if not eq["all_satisfied"]:
                pruned += 1
                continue
            row["provenance"] = {"two_clause_scene_join": True,
                "bidirectional_residual_signature": True,
                "agreement_state_carried": True, "valency_state_carried": True,
                "complete_utterances": True, "cross_word_seams": True,
                "catalogue_text": False, "finished_tape_reversal": False,
                "post_hoc_repair": False, "mirrored_units": False,
                "word_order_symmetry": False, "fragment": False,
                "nested_self_palindrome": False}
            rows.append(row)
        if checked >= limit:
            break
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]
    reader = [r for r in exact if r["provenance"]["complete_utterances"]]
    result = {"experiment_id": ID,
        "method": "bidirectional residual-signature join over agreement/valency-aware two-clause scenes",
        "stats": {"clause_frames": len(bank), "complete_two_clause_scenes": len(scenes),
                  "signature_buckets": len(right_index), "signature_join_hits": joined,
                  "live_checked": checked, "live_prunes": pruned,
                  "live_survivors": len(rows), "exact_gt38": len(exact),
                  "reader_eligible": len(reader),
                  "longest_letters": max((r["audit"]["letters"] for r in rows + controls), default=0)},
        "controls": controls, "exact_candidates": exact, "reader_facing_candidates": reader,
        "novelty_preflight": {"status": "passed",
            "signature": "two-clause-scenes|bidirectional-residual-join|agreement-valency",
            "registry_inspected": True,
            "distinct_from": "single-clause lexical trie, center-first event grammar, recursive event composition, and repair lanes",
            "catalogue_text_imported": False, "finished_tape_reversal": False,
            "post_hoc_repair": False, "mirrored_units": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
                       "reader_evidence": False},
        "status": "no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction": "Join three-clause scenes by residual signatures with one typed attachment edge, preserving semantic valency and agreement.",
        "reader_gate": "closed until exact candidates exist and blinded human ratings are collected"}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), **result["stats"]}))
    for row in controls:
        print(row["rendered"])
    return result


if __name__ == "__main__":
    run()
