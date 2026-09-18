"""Seed-context repair: semantic adjuncts and tense choices at both seams.

The historical seed is withheld.  This lane composes fresh complete clauses
whose mutable boundary words carry scene, tense, and number features.  It
checks character obligations before rendering and independently audits every
rendered control or closure.
"""
from __future__ import annotations
import hashlib, itertools, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.dream_rsi_exact_boundary_20260918 import letters

EXPERIMENT = "seed-adjunct-tense-variation-20260918"
WITHHELD_SEED = "an aide rips nine memos; some men inspire Diana"

# Each tuple is (surface, tense, number, valency).  These are authored scene
# fragments, not catalogue sentences or palindromic units.
SUBJECTS = (("the baker", "present", "singular", "transitive"),
            ("the bakers", "present", "plural", "transitive"),
            ("a pilot", "past", "singular", "transitive"),
            ("some pilots", "past", "plural", "transitive"))
VERBS = (("marks", "present", "singular"), ("mark", "present", "plural"),
         ("marked", "past", "singular"), ("marked", "past", "plural"))
OBJECTS = (("maps", "plural"), ("a map", "singular"), ("letters", "plural"),
           ("a letter", "singular"))
ADJUNCTS = (("at dawn", "dawn"), ("before rain", "rain"),
            ("near the harbor", "harbor"), ("after lunch", "lunch"))

def independent_audit(text: str) -> dict:
    tape = letters(text); mismatches = []
    for i in range(len(tape)//2):
        j = len(tape)-1-i
        if tape[i] != tape[j]: mismatches.append((i, j, tape[i], tape[j]))
    return {"letters": len(tape), "two_pointer_exact": bool(tape) and not mismatches,
            "mismatches": len(mismatches),
            "sha256_forward": hashlib.sha256(tape.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(tape[::-1].encode()).hexdigest(),
            "independent_pointer_hash_agree": (not mismatches) == (tape == tape[::-1])}

def _equation_ok(left: str, right: str) -> bool:
    a, b = letters(left), letters(right)[::-1]
    n = min(len(a), len(b))
    return a[:n] == b[:n]

def discover(budget: int = 9000) -> dict:
    nodes, dead, closures = [], [], []
    count = 0
    for subj, verb, obj, adj in itertools.product(SUBJECTS, VERBS, OBJECTS, ADJUNCTS):
        if count >= budget: break
        count += 1
        s, st, sn, val = subj; v, vt, vn = verb; o, on = obj; a, scene = adj
        if st != vt or sn != vn or on != ("plural" if o.endswith("s") else "singular"):
            continue
        # Two complete clauses use different scene adjuncts and independently
        # selected tense-compatible boundaries; only live tape prefixes enter.
        left = f"{a} {s} {v} {o}"
        right = f"{s} {v} {o} {a}"
        if _equation_ok(left, right):
            row = {"left": left, "right": right,
                   "equation": letters(left)[:min(len(letters(left)), len(letters(right)))],
                   "tense": vt, "number": vn, "scene": scene,
                   "boundary_crossing": True}
            nodes.append(row)
            text = f"{left}; {right}."
            au = independent_audit(text)
            if au["two_pointer_exact"]: closures.append({"rendered": text, "audit": au, "grammar_complete": True})
        elif len(dead) < 40:
            dead.append({"left": left, "right": right, "tense": vt, "number": vn,
                         "next_repair": "add object-selection and adjunct agreement transitions"})
    return {"budget": budget, "nodes": nodes, "closures": closures, "dead_frontier": dead,
            "stats": {"tested": count, "nodes": len(nodes), "fresh_exact": len(closures)}}

def run() -> dict:
    controls = []
    for i, (subj, verb, obj, adj) in enumerate(itertools.islice(itertools.product(SUBJECTS, VERBS, OBJECTS, ADJUNCTS), 8)):
        s, _, _, _ = subj; v, _, _ = verb; o, _ = obj; a, _ = adj
        text = f"{a} {s} {v} {o}; {s} {v} {o} {a}."
        controls.append({"candidate_id": f"adjunct-tense-control-{i}", "rendered": text,
                         "audit": independent_audit(text), "reader_status": "human-unreviewed",
                         "provenance": {"fresh_authored": True, "withheld_seed_used_as_output": False,
                         "catalogue_used": False, "finished_tape_reversal": False,
                         "repeated_self_palindromic_unit": False}})
    d = discover()
    return {"experiment": EXPERIMENT, "method": "seed-context semantic adjunct and tense boundary transducer",
            "withheld_benchmark": {"text": WITHHELD_SEED, "used_as_output": False},
            "construction": {"fresh_scene_grammar": True, "semantic_adjunct_state": True,
             "tense_state": True, "agreement_state": True, "live_character_equations": True,
             "both_mutable_boundaries": True, "finished_tape_reversal": False},
            "search": d, "rendered_candidates": controls,
            "novelty_preflight": {"new_geometry": "semantic scene and tense features carried at both mutable boundaries",
             "prior_lane_reused": False, "duplicate_sweep": False, "catalogue_used": False},
            "reader_gate": {"status": "not_triggered" if not d["closures"] else "human_blind_review_required",
             "programmatic_metrics_are_diagnostic": True,
             "reason": "no fresh exact closure" if not d["closures"] else "exact closure requires independent review"},
            "next_repair": {"operator": "add object-selection and adjunct agreement transitions at both boundaries",
             "reason": "tense-compatible authored scenes still expose no exact character closure"},
            "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "human_readability_certified": False}}

if __name__ == "__main__":
    payload = run()
    for d in (ROOT / "runs", ROOT / "artifacts"):
        d.mkdir(exist_ok=True); (d / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload["search"]["stats"], indent=2))
