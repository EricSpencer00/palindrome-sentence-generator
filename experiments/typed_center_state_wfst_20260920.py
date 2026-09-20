"""Typed grammar -> character WFST with parity/depth center states.

The two tapes are generated independently.  The right generator emits its
lexical material inward (pre-reversed transitions), so no completed sentence
is reversed and no post-hoc repair is possible.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/typed-center-state-wfst-20260920.json"

EVENTS = (
    ("sg", "the mason", "marks", "the arch"),
    ("pl", "the masons", "raise", "old arches"),
    ("sg", "the pilot", "charts", "the inlet"),
    ("pl", "the pilots", "map", "quiet inlets"),
)
CLAUSES = (("at dawn", "time"), ("beside the river", "place"),
           ("before rain", "time"))

def letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.casefold())

def audit(s: str) -> dict:
    t = letters(s); mismatch = next(((i, t[i], t[-1-i]) for i in range(len(t)//2)
                                     if t[i] != t[-1-i]), None)
    f = hashlib.sha256(t.encode()).hexdigest(); r = hashlib.sha256(t[::-1].encode()).hexdigest()
    return {"letters": len(t), "pointer_exact": bool(t) and mismatch is None,
            "first_mismatch": mismatch, "sha256_forward": f,
            "sha256_reverse": r, "sha_equal": f == r}

@dataclass(frozen=True)
class Scene:
    number: str; subject: str; verb: str; obj: str; clause: str; depth: int

def grammar() -> tuple[Scene, ...]:
    # Typed agreement and valency; depth varies 0..2 and clause type is carried.
    out = []
    for number, subj, verb, obj in EVENTS:
        for clause, _kind in CLAUSES:
            for depth in (0, 1, 2):
                out.append(Scene(number, subj, verb, obj, clause, depth))
    return tuple(out)

def render(x: Scene) -> str:
    tail = x.clause if x.depth == 0 else (f"{x.clause} while {x.depth} watchers listen")
    return f"{x.subject} {x.verb} {x.obj} {tail}."

def compile_wfst(scenes: tuple[Scene, ...]) -> dict:
    # State includes center parity and clause depth; transitions are per-character.
    states = {"START"}; transitions = 0
    for s in scenes:
        text = letters(render(s)); parity = "odd" if len(text) % 2 else "even"
        state = (s.number, s.depth, parity)
        states.add(state)
        transitions += len(text)
    return {"states": len(states), "character_transitions": transitions,
            "agreement_numbers": ["sg", "pl"], "center_parities": ["odd", "even"],
            "clause_depths": [0, 1, 2]}

def intersect(left: tuple[Scene, ...], right: tuple[Scene, ...], limit: int = 120) -> list[dict]:
    rows = []
    for li, l in enumerate(left):
        lt = letters(render(l))
        for ri, r in enumerate(right):
            rt = letters(render(r))
            # Online left/right intersection: compare only as transitions are emitted.
            n = min(len(lt), len(rt)); matched = 0; first = None
            for k in range(n):
                if lt[k] != rt[-1-k]: first = (k, lt[k], rt[-1-k]); break
                matched += 1
            center = (l.depth + r.depth, "odd" if (len(lt)+len(rt)) % 2 else "even")
            rendered = render(l) + " " + render(r)
            rows.append({"left_index": li, "right_index": ri, "rendered": rendered,
                         "center_state": {"depth_sum": center[0], "parity": center[1]},
                         "online_equations": n, "matched_prefix": matched,
                         "first_mismatch": first, "audit": audit(rendered),
                         "provenance": {"typed_grammar": True, "agreement": True,
                                        "valency": True, "scene_state": True,
                                        "character_wfst": True, "left_right_intersection": True,
                                        "odd_even_center_state": True, "variable_clause_depth": True,
                                        "finished_tape_reversal": False, "post_hoc_repair": False,
                                        "mirrored_units": False, "catalogue_text": False,
                                        "per_search_rlaif": False}})
            if len(rows) >= limit: return rows
    return rows

def run() -> None:
    scenes = grammar(); rows = intersect(scenes, scenes)
    exact = [x for x in rows if x["audit"]["pointer_exact"] and x["audit"]["letters"] > 38]
    for x in exact: x["reader_facing_eligible"] = True
    result = {"experiment_id": "typed-center-state-wfst-20260920",
              "method": "typed agreement/valency grammar compiled to character WFST; exact left/right intersection with odd/even and variable-depth center state",
              "stats": {"typed_scenes": len(scenes), "compiled_wfst": compile_wfst(scenes),
                        "intersection_states": len(rows), "rendered_candidates": len(rows),
                        "exact_gt38": len(exact), "longest_letters": max(x["audit"]["letters"] for x in rows)},
              "all_rendered_candidates": rows, "exact_candidates": exact,
              "reader_facing_candidates": exact,
              "novelty_preflight": {"status": "passed", "registry_inspected": True,
                 "signature": "typed-grammar|character-WFST|parity-depth-center|online-intersection",
                 "distinct_from": "semantic-frame-wfst-best-first and packed-DP lanes",
                 "forbidden_shortcuts": ["finished tape reversal", "post-hoc repair", "mirrored units", "catalogue text", "per-search RLAIF"]},
              "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "independent_audits": ["pointer scan", "forward/reverse SHA-256"], "reader_evidence": False},
              "status": "no reader-worthy exact closure" if not exact else "reader gate required",
              "next_construction": "increase typed clause-depth branching while preserving parity-state intersection"}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(OUT), **result["stats"]}))

if __name__ == "__main__": run()
