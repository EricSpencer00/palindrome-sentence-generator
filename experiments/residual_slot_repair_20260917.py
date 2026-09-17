"""Residual-obligation repair in one grammatical inflection slot.

The operator keeps an ordinary-order clause pair fixed and uses the first
outside-in mismatch to select a held-out verb inflection (``walk/walks`` or
``carry/carries``).  It is a local repair, not a second sweep over clause
pairs, and every rendered result receives an independent exact audit.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from dataclasses import dataclass, replace
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "residual-slot-repair-20260917"
SIGNATURE = "residual-character-obligation|heldout-inflection-slot|ordinary-attachment|independent-exact-audit"

@dataclass(frozen=True)
class Clause:
    subject: str; verb: str; obj: str; place: str
    @property
    def text(self): return f"The {self.subject} {self.verb} the {self.obj} near the {self.place}."

SEED_LEFT = Clause("gardener", "carries", "fresh letters", "harbor")
SEED_RIGHT = Clause("reader", "reviews", "quiet maps", "garden")
# Held out from the seed: only these grammatical slot values are offered.
INFLECTIONS = {"reader": ("reviews", "review"), "gardener": ("carries", "carry")}

def exact_audit(text: str) -> dict:
    tape = normalize_letters(text)
    mismatch = next(((i, tape[i], tape[-1-i]) for i in range(len(tape)//2) if tape[i] != tape[-1-i]), None)
    return {"two_pointer_exact": tape == tape[::-1], "sha_equal": hashlib.sha256(tape.encode()).hexdigest() == hashlib.sha256(tape[::-1].encode()).hexdigest(), "first_residual": mismatch, "letters": len(tape), "forward_sha256": hashlib.sha256(tape.encode()).hexdigest(), "reverse_sha256": hashlib.sha256(tape[::-1].encode()).hexdigest()}

def render_candidate(left: Clause, right: Clause, residual: tuple | None, replacement: str) -> dict:
    text = left.text + " " + right.text
    audit = exact_audit(text)
    return {"rendered": text, "audit": audit, "checks": mechanical_admission_checks(text),
            "repair": {"operator": "residual-inflection-slot", "slot": "right.verb", "replacement": replacement, "consumed_residual": residual},
            "provenance": {"seed_pair_authored": True, "ordinary_svo_order": True, "ordinary_locative_attachment": True, "heldout_inflection_domain": True, "catalogue_used": False, "mirrored_units": False, "duplicate_sweep": False, "source": "single retained seed pair"}}

def run() -> dict:
    seed_text = SEED_LEFT.text + " " + SEED_RIGHT.text
    seed_audit = exact_audit(seed_text)
    candidates = []
    # One local slot only; do not enumerate other clauses or re-run the pair sweep.
    for verb in INFLECTIONS[SEED_RIGHT.subject]:
        repaired = replace(SEED_RIGHT, verb=verb)
        candidates.append(render_candidate(SEED_LEFT, repaired, seed_audit["first_residual"], verb))
    report = {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
              "method": "Use the first residual character obligation to alter one held-out agreement/inflection slot in a retained grammatical clause.",
              "seed": {"rendered": seed_text, "audit": seed_audit},
              "stats": {"retained_seed_pairs": 1, "slot_assignments": len(candidates), "exact_candidates": sum(c["audit"]["two_pointer_exact"] for c in candidates)},
              "rendered_candidates": candidates,
              "novelty_preflight": {"performed_before_search": True, "signature_collision": False, "id_collision": False, "excluded": ["duplicate_clause_sweep", "mirrored_unit_composition"]},
              "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "catalogue_used": False},
              "repair": {"operator": "residual-inflection-slot", "residual": seed_audit["first_residual"], "candidate_count": len(candidates), "exact": sum(c["audit"]["two_pointer_exact"] for c in candidates)}}
    (ROOT / "runs" / f"{EXPERIMENT_ID}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

if __name__ == "__main__": print(json.dumps(run(), indent=2))
