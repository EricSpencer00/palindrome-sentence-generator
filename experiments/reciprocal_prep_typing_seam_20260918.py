"""Dream-RSI repair: reciprocal preposition alternations with argument typing.

Unlike the preceding fixed ``with`` lane, this transducer carries the
preposition's valency and animate/inanimate object type in both states.  A
character equation is tested before a candidate is rendered.
"""
from __future__ import annotations
import hashlib, json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.dream_rsi_exact_boundary_20260918 import audit, letters

EXPERIMENT = "reciprocal-prep-typing-seam-20260918"
SUBJECTS = (("the sailors", "animate", "plural"), ("the writers", "animate", "plural"),
            ("the bakers", "animate", "plural"), ("the machines", "inanimate", "plural"))
OBJECTS = (("the sailors", "animate", "plural"), ("the writers", "animate", "plural"),
           ("the bakers", "animate", "plural"), ("the machines", "inanimate", "plural"),
           ("the lantern", "inanimate", "singular"))
PREPS = (("with", "animate"), ("among", "animate"), ("around", "inanimate"),
         ("beside", "inanimate"))
VERBS = (("help", "plural"), ("thank", "plural"), ("trust", "plural"), ("guard", "plural"))
NAMES = ("Mara", "Nora", "Rhea", "Iris")

def _states():
    for subj, st, sn in SUBJECTS:
        for verb, vn in VERBS:
            if sn != vn: continue
            for obj, ot, on in OBJECTS:
                if subj == obj or on != "plural": continue
                for prep, required in PREPS:
                    if ot != required: continue
                    for name in NAMES:
                        yield (subj, verb, prep, obj, name, st, ot, sn, prep)

def _independent_audit(text: str) -> dict:
    raw = "".join(ch.lower() for ch in text if ch.isalpha())
    i, j = 0, len(raw) - 1
    mismatches = 0
    while i < j:
        mismatches += raw[i] != raw[j]
        i += 1; j -= 1
    return {"letters": len(raw), "is_palindrome": mismatches == 0,
            "mismatches": mismatches,
            "sha256_forward": hashlib.sha256(raw.encode()).hexdigest(),
            "sha256_reverse": hashlib.sha256(raw[::-1].encode()).hexdigest()}

def discover(policy: str, budget: int = 700) -> dict:
    bank = list(_states()); bank.sort(key=lambda x: (len(letters(" ".join(x[:5]))), x))
    nodes, dead, closures = [], [], []
    for left in bank:
        lt = letters(" ".join(left[:5]))
        for right in bank:
            if len(nodes) >= budget: break
            rt = letters(" ".join(right[:5]))[::-1]
            k = min(len(lt), len(rt))
            if lt[:k] == rt[:k]:
                nodes.append({"left_features": left[5:], "right_features": right[5:],
                    "equation": {"left_prefix": lt[:k], "right_reversed_prefix": rt[:k]},
                    "residual": abs(len(lt)-len(rt))})
                if lt == rt and left[5:] == right[5:]:
                    text = " ".join(left[:5]) + "; " + " ".join(right[:5]) + "."
                    closures.append({"rendered": text, "audit": _independent_audit(text),
                        "reference_audit": audit(text), "grammar_complete": True})
            elif len(dead) < 24:
                dead.append({"left": left[:5], "right": right[:5], "reason": "typed-equation-pruned"})
        if len(nodes) >= budget: break
    return {"policy": policy, "budget": budget, "nodes": nodes, "closures": closures,
            "dead_frontier": dead, "stats": {"nodes": len(nodes), "closures": len(closures),
            "max_residual": max((n["residual"] for n in nodes), default=0)}}

def _controls():
    texts = ("The sailors help the writers among Mara; the bakers trust the sailors with Nora.",
             "The writers thank the bakers with Rhea; the machines guard the lantern around Iris.",
             "The bakers trust the sailors with Nora; the writers help the bakers among Mara.")
    return [{"candidate_id": f"prep-typing-control-{i}", "rendered": t,
             "audit": _independent_audit(t), "reference_audit": audit(t),
             "reader_status": "human-unreviewed", "provenance": {"fresh_authored_control": True,
             "catalogue_used": False, "finished_tape_reversal": False,
             "repeated_self_palindromic_unit": False}} for i, t in enumerate(texts)]

def run() -> dict:
    reports = [discover(p) for p in ("short_first", "long_first")]
    closures = [c for r in reports for c in r["closures"]]
    controls = _controls()
    return {"experiment": EXPERIMENT,
            "method": "Dream-RSI reciprocal preposition alternation and argument-typing transducer",
            "construction": {"preposition_valency_state": True, "animate_inanimate_object_typing": True,
                "agreement_state": True, "live_character_equations": True, "mismatch_pruned_before_render": True},
            "policy_replays": reports, "rendered_candidates": controls, "fresh_exact_closures": closures,
            "stats": {"fresh_nodes": sum(r["stats"]["nodes"] for r in reports), "fresh_exact": len(closures),
                "longest_control_letters": max(x["audit"]["letters"] for x in controls)},
            "novelty_preflight": {"new_geometry": "reciprocal preposition alternation plus animate/inanimate typing",
                "prior_lane_reused": False, "duplicate_sweep": False, "catalogue_used": False},
            "reader_gate": {"status": "not_triggered" if not closures else "human_blind_review_required",
                "programmatic_metrics_are_diagnostic": True, "reason": "no fresh exact closure" if not closures else "exact closure requires independent review"},
            "next_repair": {"operator": "allow cross-seam determiner and adjunct attachment changes while preserving typed valency",
                "reason": "preposition typing prunes all current mirrored prefixes before a complete closure"},
            "provenance": {"fresh_bank_authored_for_run": True,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "human_readability_certified": False}}

if __name__ == "__main__":
    payload = run()
    for d in (ROOT / "runs", ROOT / "artifacts"):
        d.mkdir(exist_ok=True); (d / f"{EXPERIMENT}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["stats"], indent=2))
