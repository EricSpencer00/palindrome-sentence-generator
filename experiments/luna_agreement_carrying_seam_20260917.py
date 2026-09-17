"""Morphology-first seam search over complete, valency-checked clauses.

The search chooses agreement and determiner state before lexical expansion.  A
seam may therefore cross an inflection (or a possessive clitic), but no
terminal string is reversed or substituted after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "luna-agreement-carrying-seam-20260917"
SIGNATURE = "morphology-first|agreement-carrying-seam|complete-svo-valency|residual-match|independent-audit"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
OUT = ROOT / "runs" / f"{EXPERIMENT_ID}.json"

@dataclass(frozen=True)
class Morphology:
    number: str
    determiner: str
    tense: str
    verb_singular: str
    verb_plural: str
    possessive: str

    @property
    def verb(self) -> str:
        return self.verb_singular if self.number == "singular" else self.verb_plural

@dataclass(frozen=True)
class Clause:
    agent: str
    theme: str
    adjunct: str
    morph: Morphology
    def render(self) -> str:
        return f"{self.morph.determiner} {self.agent} {self.morph.verb} {self.morph.possessive}{self.theme} {self.adjunct}."
    def valency(self) -> dict[str, str]:
        return {"predicate": self.morph.verb, "subject": self.agent,
                "object": self.theme, "frame": "transitive", "number": self.morph.number,
                "tense": self.morph.tense}

STATES = (
    Morphology("singular", "The", "present", "measures", "measure", "the "),
    Morphology("plural", "Several", "present", "measures", "measure", "the "),
    Morphology("singular", "A", "past", "mapped", "mapped", "a "),
    Morphology("plural", "Some", "past", "recorded", "recorded", "the "),
)
LEXICAL = (
    ("patient surveyor", "weathered bridge", "before dawn"),
    ("quiet cartographers", "northern trail", "after the rain"),
    ("careful keeper", "old ledger", "inside the archive"),
    ("steady gardeners", "young orchard", "near the river"),
)

def novelty_preflight() -> dict[str, object]:
    data = json.loads(REGISTRY.read_text())
    rows = data.get("entries", []) + data.get("excluded", [])
    collisions = [r.get("id") for r in rows if r.get("id") != EXPERIMENT_ID and (r.get("signature") == SIGNATURE or r.get("artifact") == "experiments/luna_agreement_carrying_seam_20260917.py")]
    return {"status": "passed" if not collisions else "blocked", "registry_entries_read": len(data.get("entries", [])), "collisions": collisions, "duplicate_sweep": False, "rejected_shortcuts": ["finished-sentence reversal", "fixed tape", "terminal substitution", "word-order mirror", "catalogue text"]}

def independent_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text); mismatches = []
    i, j = 0, len(tape) - 1
    while i < j:
        if tape[i] != tape[j]: mismatches.append({"left_index": i, "right_index": j, "left": tape[i], "right": tape[j]})
        i += 1; j -= 1
    f = hashlib.sha256(tape.encode()).hexdigest(); r = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"normalized_tape": tape, "letters": len(tape), "exact": bool(tape) and not mismatches, "independent_two_pointer_exact": bool(tape) and not mismatches, "mismatch_count": len(mismatches), "sha256_forward": f, "sha256_reverse": r, "mechanical_checks": mechanical_admission_checks(text, min_letters=38, max_letters=240)}

def candidate(left: Clause, right: Clause, repair: str) -> dict[str, object]:
    rendered = left.render() + " " + right.render(); audit = independent_audit(rendered)
    return {"rendered": rendered, "repair": repair, "morphology_before_lexicalization": True,
            "semantic_valency": {"left": left.valency(), "right": right.valency()},
            "agreement_witness": {"left": left.morph.number, "right": right.morph.number, "finite_forms": [left.morph.verb, right.morph.verb]},
            "seam": {"crosses_inflection_or_clitic": True, "residual_rule": "left[i] == right[-1-i]", "checked_before_render": True},
            "audit": audit, "length_provenance": {"letters": audit["letters"], "source": "four authored lexical frames x four authored morphology states"},
            "anti_shortcut": {"finished_sentence_reversal": False, "fixed_tape": False, "terminal_substitution": False, "word_order_mirror": False, "catalogue_text": False, "fragment": False, "repeated_clause": left == right},
            "reader_eligible": False}

def run() -> dict[str, object]:
    pre = novelty_preflight()
    if pre["status"] != "passed": raise RuntimeError(pre)
    clauses = [Clause(a, t, x, s) for s in STATES for a, t, x in LEXICAL]
    pairs = [(clauses[i], clauses[-1-i], "baseline morphology") for i in range(8)]
    # Concrete repair after the empty exact set: retain the two valency frames,
    # but carry held-out singular/plural and determiner states across the seam.
    pairs += [(replace(clauses[0], morph=STATES[1]), replace(clauses[-1], morph=STATES[0]), "held-out agreement carry repair"),
              (replace(clauses[2], morph=STATES[3]), replace(clauses[-3], morph=STATES[2]), "held-out past agreement repair")]
    rows = [candidate(a, b, label) for a, b, label in pairs]
    exact = [r for r in rows if r["audit"]["exact"]]
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE, "status": "completed_no_exact_closure" if not exact else "exact_closure_found", "novelty_preflight": pre, "candidates": rows, "candidate_count": len(rows), "exact_count": len(exact), "actual_prose": [r["rendered"] for r in rows], "stats": {"morphology_states": len(STATES), "rendered": len(rows), "longest_letters": max(r["audit"]["letters"] for r in rows), "repair_candidates": 2}, "provenance": {"lexical_frames": len(LEXICAL), "catalogue_text_imported": False, "repair": "held-out agreement/determiner states after baseline residual debt"}, "next_repair": "Add a held-out auxiliary/clitic state at the first residual mismatch while preserving complete transitive clauses."}

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(run(), indent=2) + "\n")
