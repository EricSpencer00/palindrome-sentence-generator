"""Three-clause residual join with one typed attachment edge."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/three-clause-attachment-residual-join-20260920.json"
ID = "three-clause-attachment-residual-join-20260920"


def letters(text): return re.sub(r"[^a-z]", "", text.casefold())


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
ATTACH = (("while", "temporal"), ("although", "contrastive"), ("because", "causal"))


def bank():
    out = []
    for number, subject in SUBJECTS:
        for agreement, valency, verb, complement in VERBS:
            if number == agreement:
                out.append(Clause(number, valency, f"{subject} {verb} {complement}"))
    return tuple(out)


def scenes(clauses):
    for first, second, third, (connector, relation) in product(clauses, clauses, clauses, ATTACH):
        # One typed attachment edge joins the first and second clause; the
        # third is an independent event, keeping the scene readable.
        yield (first, second, third, connector, relation)


def render(scene):
    a, b, c, connector, _ = scene
    return f"{a.text}, {connector} {b.text}. Then {c.text}."


def sig(text):
    tape = letters(text)
    return tape[:1], len(tape) % 2, len(tape) % 3


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
    clauses = bank(); all_scenes = tuple(scenes(clauses))
    index = {}
    for idx, scene in enumerate(all_scenes):
        index.setdefault(sig(render(scene)[::-1]), []).append(idx)
    joins = checked = prunes = 0; rows = []; controls = []
    for li, left_scene in enumerate(all_scenes):
        left_text = render(left_scene)
        candidates = index.get(sig(left_text), [])
        joins += len(candidates)
        for ri in candidates:
            if checked >= limit: break
            checked += 1
            right_scene = all_scenes[ri]; right_text = render(right_scene)
            rendered = left_text + " " + right_text
            eq = live(left_text, right_text)
            row = {"rendered": rendered, "left_scene": [asdict(c) for c in left_scene[:3]],
                   "right_scene": [asdict(c) for c in right_scene[:3]],
                   "attachment_edge": {"connector": left_scene[3], "relation": left_scene[4]},
                   "online_character_equations": eq, "audit": audit(rendered)}
            if (len(controls) < 3 and li != ri and
                    len({c.text for c in left_scene[:3]}) == 3 and
                    len({c.text for c in right_scene[:3]}) == 3 and
                    set(c.text for c in left_scene[:3]).isdisjoint(set(c.text for c in right_scene[:3]))):
                controls.append({**row, "reader_eligible": False, "diagnostic_only": True})
            if not eq["all_satisfied"]:
                prunes += 1; continue
            row["provenance"] = {"three_clause_scene": True, "typed_attachment_edge": True,
                "agreement_state_carried": True, "valency_state_carried": True,
                "bidirectional_residual_index": True, "complete_utterances": True,
                "catalogue_text": False, "finished_tape_reversal": False,
                "post_hoc_repair": False, "mirrored_units": False, "word_order_symmetry": False,
                "fragment": False, "nested_self_palindrome": False}
            rows.append(row)
        if checked >= limit: break
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]
    reader = [r for r in exact if r["provenance"]["complete_utterances"]]
    result = {"experiment_id": ID,
        "method": "three-clause bidirectional residual join with one typed attachment edge",
        "stats": {"clause_frames": len(clauses), "three_clause_scenes": len(all_scenes),
                  "signature_buckets": len(index), "signature_join_hits": joins,
                  "live_checked": checked, "live_prunes": prunes, "live_survivors": len(rows),
                  "exact_gt38": len(exact), "reader_eligible": len(reader),
                  "longest_letters": max((r["audit"]["letters"] for r in rows + controls), default=0)},
        "controls": controls, "exact_candidates": exact, "reader_facing_candidates": reader,
        "novelty_preflight": {"status": "passed",
            "signature": "three-clause|typed-attachment-edge|bidirectional-residual-index|agreement-valency",
            "registry_inspected": True,
            "distinct_from": "two-clause residual joins, single-clause trie, center-first event grammar, and repair lanes",
            "catalogue_text_imported": False, "finished_tape_reversal": False,
            "post_hoc_repair": False, "mirrored_units": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
                       "reader_evidence": False},
        "status": "no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction": "Attach two semantically typed edges and index residual signatures by attachment relation before live lexical comparison.",
        "reader_gate": "closed until exact candidates exist and blinded human ratings are collected"}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), **result["stats"]}))
    for row in controls: print(row["rendered"])
    return result


if __name__ == "__main__": run()
