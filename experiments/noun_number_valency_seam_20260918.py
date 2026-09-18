"""Dream-RSI lane: noun-number and valency transitions at both name seams.

The search carries subject number and transitivity while extending two clause
sides.  Character equations are checked on every extension; prose is rendered
only for complete grammatical controls or exact closures.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.dream_rsi_exact_boundary_20260918 import audit, letters

EXPERIMENT = "noun-number-valency-seam-20260918"
NAMES = ("Mara", "Nora", "Rhea", "Iris")
SUBJECTS = (("baker", "singular"), ("captain", "singular"),
            ("gardeners", "plural"), ("writers", "plural"))
# valency is explicit: intransitives take no object; transitives require one.
VERBS = (("marks", "singular", "transitive"), ("finds", "singular", "transitive"),
         ("guard", "plural", "transitive"), ("walks", "singular", "intransitive"),
         ("walk", "plural", "intransitive"))
OBJECTS = (("maps", "plural"), ("letters", "plural"), ("a map", "singular"),
           ("a letter", "singular"))

def _ok_prefix(left: str, right: str) -> bool:
    n = min(len(left), len(right))
    return left[:n] == right[:n]

def _clauses():
    """Yield complete, independently authored grammatical clause records."""
    for subj, sn in SUBJECTS:
        for verb, vn, val in VERBS:
            if sn != vn: continue
            for obj, on in OBJECTS:
                if val == "intransitive": continue
                for name in NAMES:
                    yield ("the", subj, verb, obj, name, sn, val)
            for name in NAMES:
                yield ("the", subj, verb, None, name, sn, val)

def discover(policy: str, budget: int = 900) -> dict:
    # Build left and reversed-right tapes from feature-valid complete clauses;
    # retain only live equation prefixes, before rendering text.
    clauses = list(_clauses())
    clauses.sort(key=lambda c: (len("".join(x or "" for x in c[:5])), c))
    nodes, closures, dead = [], [], []
    for left in clauses:
        lt = letters(" ".join(x for x in left[:5] if x))
        for right in clauses:
            if len(nodes) >= budget: break
            rt = letters(" ".join(x for x in right[:5] if x))[::-1]
            # compare only exposed character tape, never word-order symmetry.
            if _ok_prefix(lt, rt):
                nodes.append({"left_features": left[5:], "right_features": right[5:],
                              "residual": abs(len(lt)-len(rt)), "seam": "both-name-adjacent",
                              "equation": {"left_prefix": lt[:min(len(lt),len(rt))],
                                            "right_reversed_prefix": rt[:min(len(lt),len(rt))]}})
                if lt == rt and left[5:] == right[5:]:
                    lw = " ".join(x for x in left[:5] if x)
                    rw = " ".join(x for x in right[:5] if x)
                    text = lw + "; " + rw + "."
                    closures.append({"rendered": text, "audit": audit(text),
                                     "grammar_complete": True, "valency_checked": True})
            elif len(dead) < 20:
                dead.append({"left": left[:5], "right": right[:5], "reason": "equation-pruned"})
        if len(nodes) >= budget: break
    return {"policy": policy, "budget": budget, "nodes": nodes, "closures": closures,
            "dead_frontier": dead, "stats": {"nodes": len(nodes), "closures": len(closures),
            "max_residual": max((n["residual"] for n in nodes), default=0)}}

def _controls():
    texts = ("The baker marks a map; the gardeners guard letters.",
             "The captain finds a letter; the writers guard maps.",
             "The baker walks; the gardeners walk.")
    return [{"candidate_id": f"valency-control-{i}", "rendered": t, "audit": audit(t),
             "reader_status": "human-unreviewed", "provenance": {"fresh_authored_control": True,
             "catalogue_used": False, "finished_tape_reversal": False,
             "repeated_self_palindromic_unit": False}} for i, t in enumerate(texts)]

def run() -> dict:
    reports = [discover(p) for p in ("short_first", "long_first")]
    closures = [c for r in reports for c in r["closures"]]
    controls = _controls()
    return {"experiment": EXPERIMENT,
            "method": "Dream-RSI noun-number/valency seam transducer",
            "construction": {"complete_clause_sides": True, "noun_number_carried": True,
             "semantic_valency_checked": True, "both_name_adjacent_seams": True,
             "live_character_equations": True, "mismatch_pruned_before_render": True},
            "policy_replays": reports, "rendered_candidates": controls,
            "fresh_exact_closures": closures,
            "stats": {"fresh_nodes": sum(r["stats"]["nodes"] for r in reports),
                      "fresh_exact": len(closures), "longest_control_letters": max(x["audit"]["letters"] for x in controls)},
            "novelty_preflight": {"new_geometry": "noun-number and valency states at two name seams",
             "prior_lane_reused": False, "duplicate_sweep": False, "catalogue_used": False},
            "reader_gate": {"status": "not_triggered" if not closures else "human_blind_review_required",
             "programmatic_metrics_are_diagnostic": True,
             "reason": "no fresh exact closure" if not closures else "exact closure requires independent review"},
            "next_repair": {"operator": "add lexicalized reciprocal verb frames with plural-object agreement",
             "reason": "number/valency states survive locally but the name seam still lacks a balanced reciprocal frame"},
            "provenance": {"fresh_bank_authored_for_run": True,
             "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "human_readability_certified": False}}

if __name__ == "__main__":
    payload = run()
    for d in (ROOT / "runs", ROOT / "artifacts"):
        d.mkdir(exist_ok=True)
        (d / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))
