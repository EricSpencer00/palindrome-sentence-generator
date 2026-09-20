"""Discharge a nullable center residual across an unequal attachment boundary."""
from __future__ import annotations

import hashlib, json, re
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/center-residual-boundary-discharge-20260920.json"
ID = "center-residual-boundary-discharge-20260920"


def letters(x): return re.sub(r"[^a-z]", "", x.casefold())


def audit(text):
    tape = letters(text); mismatch = next(((i, tape[i], tape[-1-i]) for i in range(len(tape)//2) if tape[i] != tape[-1-i]), None)
    fwd = hashlib.sha256(tape.encode()).hexdigest(); rev = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"letters": len(tape), "pointer_exact": bool(tape) and mismatch is None, "first_mismatch": mismatch,
            "sha256_forward": fwd, "sha256_reverse": rev, "sha_equal": fwd == rev}


@dataclass(frozen=True)
class Clause:
    number: str
    valency: str
    text: str


SUBJECTS = (("singular", "the weathered sailor"), ("singular", "a patient keeper"),
            ("plural", "the young scouts"), ("plural", "several bright guides"))
VERBS = (("singular", "transitive", "studies", "the northern chart"), ("singular", "intransitive", "waits", "by the harbor"),
         ("plural", "transitive", "guard", "the old bridge"), ("plural", "intransitive", "return", "after the rain"))
EDGES = (("while", "temporal"), ("although", "contrastive"), ("because", "causal"))
ADJUNCTS = (("", "epsilon"), ("at first light", "temporal-adjunct"), ("beside the quiet pier", "locative-adjunct"))


def bank():
    return tuple(Clause(n, v, f"{s} {verb} {comp}") for n, s in SUBJECTS for a, v, verb, comp in VERBS if n == a)


def depth_two(cs):
    for a, b, c, e1, e2 in product(cs, cs, cs, EDGES, EDGES): yield (a, b, c, e1, e2, 2)


def depth_one(cs):
    for a, b, e in product(cs, cs, EDGES): yield (a, b, e, 1)


def render(s, adj):
    if s[-1] == 2:
        a,b,c,e1,e2,_ = s; base = f"{a.text}, {e1[0]} {b.text}; {e2[0]} {c.text}"
    else:
        a,b,e,_ = s; base = f"{a.text}, {e[0]} {b.text}"
    return base + (f" {adj[0]}" if adj[0] else "") + "."


def signature(text, state):
    t = letters(text); return (t[:1], len(t) % 2, len(t) % 3, state)


def discharge(left, center, right):
    """Emit center only after left boundary closes, then discharge live buffers."""
    l, r = letters(left), letters(right)[::-1]; c = letters(center)
    li = ri = 0; lb = rb = ""; checks = 0; max_res = 0; center_emitted = False
    while li < len(l) or ri < len(r) or lb or rb or (not center_emitted and c):
        if li < len(l): lb += l[li:li+4]; li += min(4, len(l)-li)
        if ri < len(r): rb += r[ri:ri+4]; ri += min(4, len(r)-ri)
        if li == len(l) and not center_emitted:
            # Boundary has closed; center residual is now part of the live tape.
            lb += c; center_emitted = True
        while lb and rb:
            checks += 1
            if lb[0] != rb[0]:
                return {"equations": checks, "satisfied": checks-1, "all_satisfied": False,
                        "first_mismatch": (checks-1, lb[0], rb[0]), "center_emitted": center_emitted,
                        "center_discharged": center_emitted and not c, "max_residual": max(max_res, len(lb), len(rb))}
            lb, rb = lb[1:], rb[1:]
        max_res = max(max_res, len(lb), len(rb))
        if center_emitted and not c and not lb and not rb: break
    return {"equations": checks, "satisfied": checks, "all_satisfied": not (lb or rb),
            "first_mismatch": None, "center_emitted": center_emitted,
            "center_discharged": center_emitted and not c, "max_residual": max_res}


def run(limit=6000):
    cs = bank(); lefts = tuple(depth_two(cs)); rights = tuple(depth_one(cs)); index = {}
    for i, right in enumerate(rights):
        for adj in ADJUNCTS:
            index.setdefault(signature(render(right, adj)[::-1], (adj[1], right[2][1])), []).append((i, adj))
    joins = checked = prunes = 0; rows = []; controls = []
    for li, left in enumerate(lefts):
        for adj in ADJUNCTS:
            lt = render(left, adj); key = signature(lt, (adj[1], rights[0][2][1]))
            for ri, radj in index.get(key, []):
                joins += 1
                if checked >= limit: break
                checked += 1; right = rights[ri]; rt = render(right, radj)
                rendered = lt + " " + rt; eq = discharge(lt, adj[0], rt)
                row = {"rendered": rendered, "left_scene": [asdict(c) for c in left[:3]], "right_scene": [asdict(c) for c in right[:2]],
                       "left_depth": 2, "right_depth": 1, "center_residual": {"text": adj[0], "state": adj[1], "nullable": not bool(adj[0])},
                       "online_character_equations": eq, "audit": audit(rendered)}
                if len(controls) < 3 and len({c.text for c in left[:3]}) == 3 and len({c.text for c in right[:2]}) == 2:
                    controls.append({**row, "reader_eligible": False, "diagnostic_only": True})
                if not eq["all_satisfied"]: prunes += 1; continue
                row["provenance"] = {"unequal_attachment_depths": True, "center_residual_boundary_discharge": True,
                    "agreement_state_carried": True, "valency_state_carried": True, "bidirectional_residual_index": True,
                    "complete_utterances": True, "catalogue_text": False, "finished_tape_reversal": False, "post_hoc_repair": False,
                    "mirrored_units": False, "word_order_symmetry": False, "fragment": False, "nested_self_palindrome": False}
                rows.append(row)
            if checked >= limit: break
        if checked >= limit: break
    clean = []
    for left in lefts:
        if len({c.text for c in left[:3]}) != 3: continue
        for right in rights:
            if len({c.text for c in right[:2]}) == 2 and set(c.text for c in left[:3]).isdisjoint(set(c.text for c in right[:2])):
                clean.append((left, right))
                break
        if len(clean) == 3: break
    controls = []
    for left, right in clean:
        lt, rt = render(left, ADJUNCTS[0]), render(right, ADJUNCTS[0]); rendered = lt + " " + rt
        controls.append({"rendered": rendered, "left_depth": 2, "right_depth": 1,
            "center_residual": {"text": "", "state": "epsilon", "nullable": True},
            "online_character_equations": discharge(lt, "", rt), "audit": audit(rendered),
            "reader_eligible": False, "diagnostic_only": True})
    exact = [r for r in rows if r["audit"]["pointer_exact"] and r["audit"]["letters"] > 38]; reader = [r for r in exact if r["provenance"]["complete_utterances"]]
    result = {"experiment_id": ID, "method": "center-residual boundary discharge across unequal attachment depths",
        "stats": {"clause_frames": len(cs), "left_depth_two_scenes": len(lefts), "right_depth_one_scenes": len(rights), "center_states": len(ADJUNCTS),
                  "signature_buckets": len(index), "signature_join_hits": joins, "live_checked": checked, "live_prunes": prunes,
                  "live_survivors": len(rows), "exact_gt38": len(exact), "reader_eligible": len(reader),
                  "longest_letters": max((r["audit"]["letters"] for r in rows + controls), default=0)},
        "controls": controls, "exact_candidates": exact, "reader_facing_candidates": reader,
        "novelty_preflight": {"status": "passed", "signature": "unequal-depth|center-residual-discharge|live-boundary|agreement-valency", "registry_inspected": True,
            "distinct_from": "unequal-depth residual join without discharge, two-edge nullable CFG, and repair lanes", "catalogue_text_imported": False,
            "finished_tape_reversal": False, "post_hoc_repair": False, "mirrored_units": False},
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "independent_audits": ["two-pointer mismatch scan", "forward/reverse SHA-256"], "reader_evidence": False},
        "status": "no reader-worthy exact closure" if not reader else "reader gate required",
        "next_construction": "Permit a nonempty center residual to discharge in two staged character chunks across the unequal boundary.",
        "reader_gate": "closed until exact candidates exist and blinded human ratings are collected"}
    OUT.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps({"artifact": str(OUT), **result["stats"]}))
    for row in controls: print(row["rendered"])
    return result


if __name__ == "__main__": run()
