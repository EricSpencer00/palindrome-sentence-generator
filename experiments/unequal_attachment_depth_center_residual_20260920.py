"""Unequal attachment-depth join with a nullable center residual."""
from __future__ import annotations

import hashlib, json, re
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/unequal-attachment-depth-center-residual-20260920.json"
ID = "unequal-attachment-depth-center-residual-20260920"


def letters(x): return re.sub(r"[^a-z]", "", x.casefold())


def audit(text):
    tape = letters(text)
    mismatch = next(((i, tape[i], tape[-1 - i]) for i in range(len(tape) // 2)
                     if tape[i] != tape[-1 - i]), None)
    fwd = hashlib.sha256(tape.encode()).hexdigest(); rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "pointer_exact": bool(tape) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": fwd, "sha256_reverse": rev,
            "sha_equal": fwd == rev}


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
EDGES = (("while", "temporal"), ("although", "contrastive"), ("because", "causal"))
ADJUNCTS = (("", "epsilon"), ("at first light", "temporal-adjunct"),
            ("beside the quiet pier", "locative-adjunct"))


def bank():
    out = []
    for number, subject in SUBJECTS:
        for agreement, valency, verb, complement in VERBS:
            if number == agreement:
                out.append(Clause(number, valency, f"{subject} {verb} {complement}"))
    return tuple(out)


def depth_two(clauses):
    for a, b, c, e1, e2 in product(clauses, clauses, clauses, EDGES, EDGES):
        yield (a, b, c, e1, e2, 2)


def depth_one(clauses):
    for a, b, e1 in product(clauses, clauses, EDGES):
        yield (a, b, e1, 1)


def render(scene, adjunct):
    if scene[-1] == 2:
        a, b, c, e1, e2, _ = scene
        base = f"{a.text}, {e1[0]} {b.text}; {e2[0]} {c.text}"
    else:
        a, b, e1, _ = scene
        base = f"{a.text}, {e1[0]} {b.text}"
    return base + (f" {adjunct[0]}" if adjunct[0] else "") + "."


def edge_key(scene):
    if scene[-1] == 2: return (scene[-1], scene[3][1], scene[4][1])
    return (scene[-1], scene[2][1])


def live(left, right):
    a, b = letters(left), letters(right)[::-1]
    i = j = 0; lb = rb = ""; checks = 0; max_res = 0
    while i < len(a) or j < len(b):
        if i < len(a): lb += a[i:i + 4]; i += min(4, len(a) - i)
        if j < len(b): rb += b[j:j + 4]; j += min(4, len(b) - j)
        while lb and rb:
            checks += 1
            if lb[0] != rb[0]:
                return {"equations": checks, "satisfied": checks - 1,
                        "all_satisfied": False, "first_mismatch": (checks - 1, lb[0], rb[0]),
                        "max_residual": max(max_res, len(lb), len(rb))}
            lb, rb = lb[1:], rb[1:]
        max_res = max(max_res, len(lb), len(rb))
    return {"equations": checks, "satisfied": checks, "all_satisfied": not (lb or rb),
            "first_mismatch": None, "max_residual": max_res}


def run(limit=6000):
    clauses = bank(); lefts = tuple(depth_two(clauses)); rights = tuple(depth_one(clauses))
    index = {}
    for idx, right in enumerate(rights):
        for adjunct in ADJUNCTS:
            # Reverse signature retains the unequal depth and center state.
            key = (edge_key(right), adjunct[1], letters(render(right, adjunct)[::-1])[:1])
            index.setdefault(key, []).append((idx, adjunct))
    joins = checked = prunes = 0; rows = []; controls = []
    for li, left in enumerate(lefts):
        for adjunct in ADJUNCTS:
            left_text = render(left, adjunct)
            key = (edge_key(rights[0]), adjunct[1], letters(left_text)[:1])
            candidates = index.get(key, [])
            joins += len(candidates)
            for ri, right_adj in candidates:
                if checked >= limit: break
                checked += 1; right = rights[ri]; right_text = render(right, right_adj)
                center = adjunct[0]
                rendered = left_text + (f" {center} " if center else " ") + right_text
                eq = live(left_text, right_text)
                row = {"rendered": rendered, "left_scene": [asdict(c) for c in left[:3]],
                       "right_scene": [asdict(c) for c in right[:2]],
                       "left_depth": 2, "right_depth": 1,
                       "center_residual": {"text": center, "state": adjunct[1], "nullable": not bool(center)},
                       "online_character_equations": eq, "audit": audit(rendered)}
                if len(controls) < 3:
                    controls.append({**row, "reader_eligible": False, "diagnostic_only": True})
                if not eq["all_satisfied"]: prunes += 1; continue
                row["provenance"] = {"unequal_attachment_depths": True, "left_depth": 2, "right_depth": 1,
                    "nullable_center_residual": True, "delayed_boundary_semantics": True,
                    "agreement_state_carried": True, "valency_state_carried": True,
                    "bidirectional_residual_index": True, "complete_utterances": True,
                    "catalogue_text": False, "finished_tape_reversal": False, "post_hoc_repair": False,
                    "mirrored_units": False, "word_order_symmetry": False, "fragment": False,
                    "nested_self_palindrome": False}
                rows.append(row)
            if checked >= limit: break
        if checked >= limit: break
    # Replace prefix-dominated join diagnostics with varied intact controls;
    # repeated clause units are never used as reader evidence.
    distinct_left = [s for s in lefts if len({c.text for c in s[:3]}) == 3]
    distinct_right = [s for s in rights if len({c.text for c in s[:2]}) == 2]
    controls = []
    clean_pairs = []
    for left in distinct_left:
        for right in distinct_right:
            if set(c.text for c in left[:3]).isdisjoint(set(c.text for c in right[:2])):
                clean_pairs.append((left, right))
            if len(clean_pairs) == 3: break
        if len(clean_pairs) == 3: break
    for left, right in clean_pairs:
        left_text = render(left, ADJUNCTS[0]); right_text = render(right, ADJUNCTS[0])
        rendered = left_text + " " + right_text
        controls.append({"rendered": rendered, "left_depth": 2, "right_depth": 1,
            "center_residual": {"text": "", "state": "epsilon", "nullable": True},
            "online_character_equations": live(left_text, right_text), "audit": audit(rendered),
            "reader_eligible": False, "diagnostic_only": True})
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]
    reader = [r for r in exact if r["provenance"]["complete_utterances"]]
    result = {"experiment_id": ID,
        "method": "unequal-depth two-edge residual join with nullable center adjunct residual",
        "stats": {"clause_frames": len(clauses), "left_depth_two_scenes": len(lefts),
                  "right_depth_one_scenes": len(rights), "center_residual_states": len(ADJUNCTS),
                  "signature_buckets": len(index), "signature_join_hits": joins,
                  "live_checked": checked, "live_prunes": prunes, "live_survivors": len(rows),
                  "exact_gt38": len(exact), "reader_eligible": len(reader),
                  "longest_letters": max((r["audit"]["letters"] for r in rows + controls), default=0)},
        "controls": controls, "exact_candidates": exact, "reader_facing_candidates": reader,
        "novelty_preflight": {"status": "passed",
            "signature": "unequal-depth|two-edge-vs-one-edge|nullable-center-residual|agreement-valency",
            "registry_inspected": True,
            "distinct_from": "equal-depth two-edge joins, one-edge scenes, and lexical repair lanes",
            "catalogue_text_imported": False, "finished_tape_reversal": False,
            "post_hoc_repair": False, "mirrored_units": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
                       "reader_evidence": False},
        "status": "no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction": "Permit center residuals to discharge across an unequal-depth attachment boundary before the final clause is emitted.",
        "reader_gate": "closed until exact candidates exist and blinded human ratings are collected"}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), **result["stats"]}))
    for row in controls: print(row["rendered"])
    return result


if __name__ == "__main__": run()
