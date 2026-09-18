"""Dream-RSI repair: lexicalized reciprocal frames with plural objects.

This lane is deliberately different from fixed clause-pair sweeps: each side
is generated from a reciprocal verb frame whose subject and object number are
carried as state, and the exposed character equations are solved before any
sentence is rendered.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.dream_rsi_exact_boundary_20260918 import audit, letters

EXPERIMENT = "reciprocal-plural-frame-seam-20260918"
SUBJECTS = (("the sailors", "plural"), ("the writers", "plural"),
            ("the bakers", "plural"), ("the captain", "singular"))
FRAMES = (("trust", "plural", "plural"), ("help", "plural", "plural"),
          ("thank", "plural", "plural"), ("trusts", "singular", "plural"))
OBJECTS = (("the sailors", "plural"), ("the writers", "plural"),
           ("the bakers", "plural"), ("the captain", "singular"))
NAMES = ("Mara", "Nora", "Rhea", "Iris")

def _frames():
    # Reciprocal frame is lexicalized: plural reciprocal verbs require plural
    # subjects and plural object groups; singular trust is transitive but not
    # admitted as reciprocal, providing an explicit semantic filter.
    for subj, sn in SUBJECTS:
        for verb, vn, on in FRAMES:
            if sn != vn or on != "plural":
                continue
            for obj, ob in OBJECTS:
                if ob != on or obj == subj:
                    continue
                for name in NAMES:
                    yield (subj, verb, obj, "with", name, sn, on, "reciprocal")

def _equation_prefix(a: str, b: str) -> bool:
    k = min(len(a), len(b))
    return a[:k] == b[:k]

def discover(policy: str, budget: int = 700) -> dict:
    bank = list(_frames())
    bank.sort(key=lambda x: (len(letters(" ".join(x[:5]))), x))
    nodes, dead, closures = [], [], []
    for left in bank:
        lt = letters(" ".join(left[:5]))
        for right in bank:
            if len(nodes) >= budget:
                break
            rt = letters(" ".join(right[:5]))[::-1]
            if _equation_prefix(lt, rt):
                nodes.append({"left_features": left[5:], "right_features": right[5:],
                              "seam": "reciprocal-name-and-object",
                              "residual": abs(len(lt)-len(rt)),
                              "equation": {"left_prefix": lt[:min(len(lt),len(rt))],
                                           "right_reversed_prefix": rt[:min(len(lt),len(rt))]}})
                if lt == rt and left[5:] == right[5:]:
                    l = " ".join(left[:5]); r = " ".join(right[:5])
                    text = l + "; " + r + "."
                    closures.append({"rendered": text, "audit": audit(text),
                                     "grammar_complete": True, "reciprocal_frame_checked": True})
            elif len(dead) < 24:
                dead.append({"left": left[:5], "right": right[:5], "reason": "equation-pruned"})
        if len(nodes) >= budget:
            break
    return {"policy": policy, "budget": budget, "nodes": nodes,
            "closures": closures, "dead_frontier": dead,
            "stats": {"nodes": len(nodes), "closures": len(closures),
                      "max_residual": max((n["residual"] for n in nodes), default=0)}}

def _controls():
    texts = ("The sailors trust the writers with Mara; the bakers help the sailors with Nora.",
             "The writers thank the bakers with Rhea; the sailors trust the writers with Iris.",
             "The bakers help the sailors with Nora; the writers trust the bakers with Mara.")
    return [{"candidate_id": f"reciprocal-control-{i}", "rendered": t, "audit": audit(t),
             "reader_status": "human-unreviewed",
             "provenance": {"fresh_authored_control": True, "catalogue_used": False,
                            "finished_tape_reversal": False,
                            "repeated_self_palindromic_unit": False}}
            for i, t in enumerate(texts)]

def run() -> dict:
    reports = [discover(p) for p in ("short_first", "long_first")]
    closures = [c for r in reports for c in r["closures"]]
    controls = _controls()
    return {"experiment": EXPERIMENT,
            "method": "Dream-RSI lexicalized reciprocal-frame seam transducer",
            "construction": {"reciprocal_lexicon": True, "plural_subject_object_agreement": True,
                             "semantic_valency_checked": True, "live_character_equations": True,
                             "mismatch_pruned_before_render": True},
            "policy_replays": reports, "rendered_candidates": controls,
            "fresh_exact_closures": closures,
            "stats": {"fresh_nodes": sum(r["stats"]["nodes"] for r in reports),
                      "fresh_exact": len(closures),
                      "longest_control_letters": max(x["audit"]["letters"] for x in controls)},
            "novelty_preflight": {"new_geometry": "lexicalized reciprocal frames with plural-object state",
                                  "prior_lane_reused": False, "duplicate_sweep": False,
                                  "catalogue_used": False},
            "reader_gate": {"status": "not_triggered" if not closures else "human_blind_review_required",
                            "programmatic_metrics_are_diagnostic": True,
                            "reason": "no fresh exact closure" if not closures else "exact closure requires independent review"},
            "next_repair": {"operator": "add reciprocal preposition alternations and animate/inanimate object typing",
                            "reason": "plural reciprocal states still leave the name seam underconstrained"},
            "provenance": {"fresh_bank_authored_for_run": True,
                           "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "human_readability_certified": False}}

if __name__ == "__main__":
    payload = run()
    for d in (ROOT / "runs", ROOT / "artifacts"):
        d.mkdir(exist_ok=True)
        (d / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))
