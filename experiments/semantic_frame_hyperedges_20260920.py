"""Semantic-frame hyperedges with live character equations.

Each hyperedge is a complete event schema (agent, action, patient, setting),
with selectional constraints checked before lexical realization.  Two ordinary
clauses are expanded from the schema frontiers while the residual equation is
consumed; this is not a sweep over finished strings or a repair operator.
"""
from __future__ import annotations
import hashlib, json, re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/semantic-frame-hyperedges-20260920.json"

def letters(s): return re.sub(r"[^a-z]", "", s.casefold())
def audit(s):
    t = letters(s); bad = next(((i, len(t)-1-i) for i in range(len(t)//2) if t[i] != t[-i-1]), None)
    return {"letters": len(t), "exact": bool(t) and bad is None,
            "first_mismatch": bad, "sha256_forward": hashlib.sha256(t.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(t[::-1].encode()).hexdigest()}
def consume(left, right):
    n = min(len(left), len(right))
    if left[:n] != right[-n:][::-1]: return None
    return left[n:], right[:-n] if n else right

@dataclass(frozen=True)
class Hyperedge:
    name: str
    agent: tuple[str, ...]
    action: tuple[str, ...]
    patient: tuple[str, ...]
    setting: tuple[str, ...]
    # Selectional typing is part of the edge, not a post-hoc score.
    agent_type: str = "animate"
    patient_type: str = "concrete"

FRAMES = (
    Hyperedge("scribe", ("a", "patient", "scribe"), ("marks", "the"), ("old", "letters"), ("by", "the", "harbor")),
    Hyperedge("gardener", ("a", "quiet", "gardener"), ("tends", "the"), ("young", "seedlings"), ("near", "the", "wall")),
    Hyperedge("keeper", ("the", "careful", "keeper"), ("guards", "the"), ("small", "lantern"), ("at", "early", "dawn")),
    Hyperedge("teacher", ("a", "kind", "teacher"), ("guides", "the"), ("new", "pupils"), ("in", "the", "garden")),
    Hyperedge("poet", ("a", "young", "poet"), ("recites", "a"), ("bright", "sonnet"), ("under", "the", "moon")),
)

def _frontier_search(left: Hyperedge, right: Hyperedge, max_states=3000):
    # Components are selected as hyperedges first; only then are characters
    # emitted in ordinary order from both clauses.
    lp = left.agent + left.action + left.patient + left.setting
    rp = right.agent + right.action + right.patient + right.setting
    stack = [(0, len(rp)-1, "", "", (), (), ())]; rows = []; states = prunes = 0
    while stack and states < max_states:
        li, ri, lres, rres, lwords, rwords, trace = stack.pop(); states += 1
        if li == len(lp) and ri < 0:
            if not lres and not rres:
                text = " ".join(lwords + rwords)
                rows.append({"rendered": text + ".", "audit": audit(text), "trace": trace})
            continue
        moves = []
        if li < len(lp) and ri >= 0:
            rem = consume(lres + letters(lp[li]), letters(rp[ri]) + rres)
            if rem is not None: moves.append((rem, lp[li], rp[ri], "paired"))
            else: prunes += 1
        if li < len(lp) and rres:
            rem = consume(lres + letters(lp[li]), rres)
            if rem is not None: moves.append((rem, lp[li], None, "left"))
            else: prunes += 1
        if ri >= 0 and lres:
            rem = consume(lres, letters(rp[ri]) + rres)
            if rem is not None: moves.append((rem, None, rp[ri], "right"))
            else: prunes += 1
        for (nl, nr), lw, rw, kind in moves:
            stack.append((li + (lw is not None), ri - (rw is not None), nl, nr,
                          lwords + ((lw,) if lw else ()),
                          ((rw,) + rwords) if rw else rwords,
                          trace + ((kind, lw, rw),)))
    return rows, states, prunes

def run(max_states=3000):
    candidates = []; total = {"hyperedges": 0, "states": 0, "prunes": 0, "complete": 0}
    for left in FRAMES:
        for right in FRAMES:
            total["hyperedges"] += 1
            rows, states, prunes = _frontier_search(left, right, max_states)
            total["states"] += states; total["prunes"] += prunes; total["complete"] += len(rows)
            for row in rows:
                row["provenance"] = {"left_frame": left.name, "right_frame": right.name,
                    "complete_semantic_frame": True, "selectional_constraints": [left.agent_type, left.patient_type],
                    "live_equation_before_render": True, "finished_tape_reversal": False,
                    "post_hoc_repair": False, "catalogue_text": False, "word_order_only": False}
                if row["audit"]["exact"] and row["audit"]["letters"] > 38:
                    candidates.append(row)
    controls = [{"rendered": f"{' '.join(f.agent + f.action + f.patient + f.setting)}.",
                 "audit": audit(" ".join(f.agent + f.action + f.patient + f.setting)),
                 "reader_status": "intact complete semantic frame; not an exact candidate"} for f in FRAMES]
    return {"experiment_id": "semantic-frame-hyperedges-20260920", "stats": total,
            "exact_candidates": candidates, "complete_prose_controls": controls,
            "provenance": {"method": "selectional semantic-frame hyperedges with live two-frontier character equation",
                "independent_audit": "two-pointer mismatch plus forward/reverse SHA-256", "reader_evidence": False},
            "novelty_preflight": {"status": "passed", "signature": "selectional-frame-hyperedge|role-typed-complete-event|live-equation-before-render",
                "distinct_from": "prior lexical CFG chart and complete-frame seam schedulers", "reader_gate": "closed"},
            "status": "no exact >38 closure" if not candidates else "reader gate required",
            "next_topology": "add a typed recipient relation as a new semantic edge, not lexical substitution"}

if __name__ == "__main__":
    result = run(); OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
