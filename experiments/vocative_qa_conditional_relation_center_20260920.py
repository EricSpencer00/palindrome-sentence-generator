"""Conditional-scope relation center over the role-typed QA clarification CSP."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/vocative_qa_relation_center_clarification_20260920.py"
spec = importlib.util.spec_from_file_location("qa_relation_base", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# Registry preflight found no vocative conditional-scope relation lane.
base.FRAMES = (
    {"speaker": "sailor", "role": "traveler", "relation": "if", "voc": ("sailor",),
     "q": ("did", "you", "see", "the", "harbor"), "a": ("the", "sailor", "saw", "the", "harbor"),
     "clar": ("the", "keeper", "confirmed"), "tail": ("then", "the", "harbor", "was", "quiet")},
    {"speaker": "keeper", "role": "witness", "relation": "since", "voc": ("keeper",),
     "q": ("did", "you", "guard", "the", "lantern"), "a": ("the", "keeper", "guarded", "the", "lantern"),
     "clar": ("the", "sailor", "agreed"), "tail": ("the", "lantern", "was", "bright")},
    {"speaker": "poet", "role": "observer", "relation": "when", "voc": ("poet",),
     "q": ("did", "you", "remember", "the", "garden"), "a": ("the", "poet", "remembered", "the", "garden"),
     "clar": ("the", "keeper", "noted"), "tail": ("the", "garden", "was", "peaceful")},
)

result = base.run()
result.update({
    "method": "vocative-qa-conditional-relation-center-20260920",
    "provenance": "fresh conditional-scope relation center (if/since/when) over complete role-typed vocative QA clarification; three residual chunks and cross-word seam remain live; no relation inventory sweep, reversal, repair, catalogue text, or mirrored units",
    "novelty_preflight": {
        "passed": True,
        "overlaps_checked": ["vocative-qa-relation-center-clarification-20260920", "relation-conditioned-voice-grammar-20260920", "center-two-relation-scope-20260920"],
        "unused_dimension": "conditional-scope relation center with role-specific connective/clarification semantics",
        "reason": "registry contains causal, concessive, voice, and dual-scope relations, but no vocative QA conditional center carrying an if/since/when scope into the clarification chunks",
    },
    "next_construction": "hold out a counterfactual unless/then clarification with explicit scope agreement; do not widen the current connective set",
})
out = ROOT / "runs/vocative-qa-conditional-relation-center-20260920.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ("method", "states", "character_prunes", "semantic_prunes", "exact_candidate_count")}))
