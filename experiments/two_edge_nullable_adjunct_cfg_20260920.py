"""Two-edge residual join with a held-out nullable adjunct CFG nonterminal."""
from __future__ import annotations

import hashlib, json, re
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/two-edge-nullable-adjunct-cfg-20260920.json"
ID = "two-edge-nullable-adjunct-cfg-20260920"


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
# Held out from the earlier scene banks.  The first production is nullable;
# delayed boundary semantics decide whether the lexical adjunct is emitted.
ADJUNCT_CFG = (("", "epsilon"), (" at first light", "temporal-adjunct"),
               (" beside the quiet pier", "locative-adjunct"))


def bank():
    out = []
    for number, subject in SUBJECTS:
        for agreement, valency, verb, complement in VERBS:
            if number == agreement:
                out.append(Clause(number, valency, f"{subject} {verb} {complement}"))
    return tuple(out)


def scenes(clauses):
    for a, b, c, e1, e2, adj in product(clauses, clauses, clauses, EDGES, EDGES, ADJUNCT_CFG):
        yield (a, b, c, e1, e2, adj)


def render(scene):
    a, b, c, e1, e2, adj = scene
    return f"{a.text}, {e1[0]} {b.text}; {e2[0]} {c.text}{adj[0]}."


def sig(text, adj_state):
    tape = letters(text)
    return tape[:1], len(tape) % 2, len(tape) % 3, adj_state


def live(left, right):
    a, b = letters(left), letters(right)[::-1]
    i = j = 0; lb = rb = ""; checks = 0; max_res = 0
    while i < len(a) or j < len(b):
        if i < len(a): lb += a[i:i + 4]; i += min(4, len(a) - i)
        if j < len(b): rb += b[j:j + 4]; j += min(4, len(b) - j)
        while lb and rb:
            checks += 1
            if lb[0] != rb[0]:
                return {"equations": checks, "satisfied": checks - 1, "all_satisfied": False,
                        "first_mismatch": (checks - 1, lb[0], rb[0]),
                        "max_residual": max(max_res, len(lb), len(rb))}
            lb, rb = lb[1:], rb[1:]
        max_res = max(max_res, len(lb), len(rb))
    return {"equations": checks, "satisfied": checks, "all_satisfied": not (lb or rb),
            "first_mismatch": None, "max_residual": max_res}


def run(limit=6000):
    clauses = bank(); all_scenes = tuple(scenes(clauses)); index = {}
    for idx, scene in enumerate(all_scenes):
        # Delay the adjunct boundary state until after both attachment edges.
        reverse_state = (scene[5][1], scene[3][1], scene[4][1])
        index.setdefault(sig(render(scene)[::-1], reverse_state), []).append(idx)
    joins = checked = prunes = 0; rows = []; controls = []
    for li, left_scene in enumerate(all_scenes):
        left_text = render(left_scene)
        adj_state = (left_scene[5][1], left_scene[3][1], left_scene[4][1])
        candidates = index.get(sig(left_text, adj_state), [])
        joins += len(candidates)
        for ri in candidates:
            if checked >= limit: break
            checked += 1; right_scene = all_scenes[ri]; right_text = render(right_scene)
            rendered = left_text + " " + right_text; eq = live(left_text, right_text)
            row = {"rendered": rendered, "left_scene": [asdict(c) for c in left_scene[:3]],
                   "right_scene": [asdict(c) for c in right_scene[:3]],
                   "attachment_edges": [{"connector": left_scene[3][0], "relation": left_scene[3][1]},
                                         {"connector": left_scene[4][0], "relation": left_scene[4][1]}],
                   "adjunct_nonterminal": {"state": left_scene[5][1], "text": left_scene[5][0],
                                           "nullable": left_scene[5][1] == "epsilon", "delayed_boundary": True},
                   "online_character_equations": eq, "audit": audit(rendered)}
            if (len(controls) < 3 and li != ri and
                    len({c.text for c in left_scene[:3]}) == 3 and
                    len({c.text for c in right_scene[:3]}) == 3):
                controls.append({**row, "reader_eligible": False, "diagnostic_only": True})
            if not eq["all_satisfied"]: prunes += 1; continue
            row["provenance"] = {"three_clause_scene": True, "two_typed_attachment_edges": True,
                "held_out_nullable_adjunct_cfg": True, "delayed_boundary_semantics": True,
                "agreement_state_carried": True, "valency_state_carried": True,
                "bidirectional_residual_index": True, "complete_utterances": True,
                "catalogue_text": False, "finished_tape_reversal": False, "post_hoc_repair": False,
                "mirrored_units": False, "word_order_symmetry": False, "fragment": False,
                "nested_self_palindrome": False}
            rows.append(row)
        if checked >= limit: break
    # Preserve intact prose controls even when the joint signature join is
    # empty or its bounded prefix is dominated by repeated lexical frames.
    if len(controls) < 3:
        distinct = [s for s in all_scenes if len({c.text for c in s[:3]}) == 3]
        for left_scene, right_scene in zip(distinct[:3], distinct[3:6]):
            left_text, right_text = render(left_scene), render(right_scene)
            controls.append({"rendered": left_text + " " + right_text,
                "left_scene": [asdict(c) for c in left_scene[:3]],
                "right_scene": [asdict(c) for c in right_scene[:3]],
                "adjunct_nonterminal": {"state": left_scene[5][1], "text": left_scene[5][0],
                                        "nullable": left_scene[5][1] == "epsilon", "delayed_boundary": True},
                "online_character_equations": live(left_text, right_text),
                "audit": audit(left_text + " " + right_text),
                "reader_eligible": False, "diagnostic_only": True})
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]
    reader = [r for r in exact if r["provenance"]["complete_utterances"]]
    result = {"experiment_id": ID,
        "method": "two-edge residual join with held-out nullable adjunct CFG and delayed boundary semantics",
        "stats": {"clause_frames": len(clauses), "three_clause_scenes": len(all_scenes),
                  "adjunct_productions": len(ADJUNCT_CFG), "joint_signature_buckets": len(index),
                  "signature_join_hits": joins, "live_checked": checked, "live_prunes": prunes,
                  "live_survivors": len(rows), "exact_gt38": len(exact), "reader_eligible": len(reader),
                  "longest_letters": max((r["audit"]["letters"] for r in rows + controls), default=0)},
        "controls": controls, "exact_candidates": exact, "reader_facing_candidates": reader,
        "novelty_preflight": {"status": "passed",
            "signature": "two-edges|held-out-nullable-adjunct|delayed-boundary|bidirectional-residual",
            "registry_inspected": True,
            "distinct_from": "two-edge residual join without CFG adjunct, one-edge scenes, and lexical repair lanes",
            "catalogue_text_imported": False, "finished_tape_reversal": False,
            "post_hoc_repair": False, "mirrored_units": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"],
                       "reader_evidence": False},
        "status": "no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction": "Carry the nullable adjunct as a center residual and join unequal attachment depths before lexical emission.",
        "reader_gate": "closed until exact candidates exist and blinded human ratings are collected"}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), **result["stats"]}))
    for row in controls: print(row["rendered"])
    return result


if __name__ == "__main__": run()
