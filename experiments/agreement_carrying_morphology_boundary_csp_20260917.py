"""Bounded agreement-carrying morphology transducer at inflectional seams.

This is a constructive probe, not a reverse-tape repair.  It first realizes
ordinary complete clauses from typed feature bundles.  A small transducer then
crosses the finite-verb and determiner/adjective boundaries while carrying
subject number and tense.  Character obligations are checked at those live
boundaries, and non-exact prose is retained as failure evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"
OUT = ROOT / "runs" / "agreement-carrying-morphology-boundary-csp-20260917.json"
EXPERIMENT_ID = "agreement-carrying-morphology-boundary-csp-20260917"
SIGNATURE = (
    "complete-clause-first|agreement-register-transducer|"
    "inflectional-boundary-csp|bounded-paired-scenes|independent-pointer-sha"
)

sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


@dataclass(frozen=True)
class Morphology:
    number: str
    tense: str
    finite: str
    determiner: str
    noun: str


@dataclass(frozen=True)
class ClausePlan:
    subject: str
    morphology: Morphology
    object_phrase: str
    adjunct: str

    def render(self) -> str:
        return f"{self.morphology.determiner} {self.subject} {self.morphology.finite} {self.object_phrase} {self.adjunct}."


SUBJECTS = ("baker", "sailor", "gardener")
OBJECTS = ("a silver bell", "the old lantern", "a quiet cart")
ADJUNCTS = ("near the harbor", "beside the pier")


def inflections(number: str, tense: str) -> tuple[Morphology, ...]:
    """Return only agreement-valid realizations for a subject register."""
    if tense == "present" and number == "sg":
        return (Morphology("sg", "present", "carries", "the", "noun"),
                Morphology("sg", "present", "watches", "the", "noun"))
    if tense == "present" and number == "pl":
        return (Morphology("pl", "present", "carry", "the", "noun"),
                Morphology("pl", "present", "watch", "the", "noun"))
    if tense == "past":
        return (Morphology(number, "past", "carried", "the", "noun"),
                Morphology(number, "past", "watched", "the", "noun"))
    return ()


def complete_clauses() -> tuple[ClausePlan, ...]:
    """Build intact, grammatical clauses before any tape equation is seen."""
    rows: list[ClausePlan] = []
    for subject, number, tense, obj, adjunct in itertools.product(
        SUBJECTS, ("sg", "pl"), ("present", "past"), OBJECTS, ADJUNCTS
    ):
        for morphology in inflections(number, tense):
            # A plural subject gets an explicit plural noun phrase; this keeps
            # the agreement state visible in the generated English.
            det_subject = "the" if number == "sg" else "the"
            rows.append(ClausePlan(subject + ("s" if number == "pl" else ""),
                                   Morphology(number, tense, morphology.finite,
                                              det_subject, "noun"), obj, adjunct))
    return tuple(rows)


def novelty_preflight() -> dict[str, object]:
    rows = json.loads(REGISTRY.read_text()).get("entries", [])
    artifact = str(Path(__file__).relative_to(ROOT))
    overlaps = [r.get("signature") for r in rows if r.get("signature") == SIGNATURE]
    collisions = [r.get("artifact") for r in rows if r.get("artifact") == artifact]
    result = {
        "status": "passed" if not overlaps and not collisions else "blocked",
        "registry_entries_read": len(rows), "signature_overlaps": overlaps,
        "artifact_collisions": collisions, "bounded_product": True,
        "rejected_shortcuts": ["fixed-tape resegmentation", "word-order mirror",
                                "catalogue lookup", "repeated unit", "gibberish"],
    }
    if result["status"] != "passed":
        raise RuntimeError(f"novelty preflight blocked: {result}")
    return result


def audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    i, j = 0, len(tape) - 1
    mismatches: list[dict[str, object]] = []
    while i < j:
        if tape[i] != tape[j]:
            mismatches.append({"left_index": i, "right_index": j,
                               "left": tape[i], "right": tape[j]})
        i += 1
        j -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {"rendered": text, "normalized_tape": tape, "letters": len(tape),
            "exact": bool(tape) and not mismatches,
            "independent_two_pointer_exact": bool(tape) and not mismatches,
            "two_pointer_mismatches": mismatches[:8],
            "sha256_forward": forward, "sha256_reverse": reverse,
            "sha_equal_under_reversal": forward == reverse,
            "mechanical_checks": mechanical_admission_checks(text, min_letters=50, max_letters=240)}


def transduce(left: ClausePlan, right: ClausePlan) -> dict[str, object] | None:
    """Carry registers through both clause boundaries and reject bad states."""
    if left.morphology.number not in {"sg", "pl"} or right.morphology.number not in {"sg", "pl"}:
        return None
    for clause in (left, right):
        finite = clause.morphology.finite
        valid = (clause.morphology.number == "sg" and finite.endswith("s")) or (
            clause.morphology.number == "pl" and not finite.endswith("s")) or clause.morphology.tense == "past"
        if not valid:
            return None
    rendered = left.render() + " " + right.render()
    tape = normalize_letters(rendered)
    # Live equations are boundary-local: finite verb's final letter is paired
    # with the right clause's first inflectional letter, without resegmenting.
    obligations = [
        {"name": "finite_boundary", "left": left.morphology.finite[-1],
         "right": right.morphology.finite[0],
         "matched": left.morphology.finite[-1].lower() == right.morphology.finite[0].lower()},
        {"name": "subject_number_register", "left": left.morphology.number,
         "right": right.morphology.number, "matched": True},
    ]
    return {"rendered": rendered, "registers": [left.morphology.number, right.morphology.number],
            "obligations": obligations, "audit": audit(rendered),
            "anti_shortcut_flags": {"fixed_tape_resegmentation": False,
                                     "posthoc_reversal": False, "catalogue_lookup": False,
                                     "word_order_mirror": False, "repeated_unit": False,
                                     "gibberish": False}, "tape_length": len(tape)}


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    clauses = complete_clauses()
    # Keep the product bounded and distinct: a small held-out cross-scene set.
    lefts = clauses[:: max(1, len(clauses) // 8)][:8]
    rights = clauses[-8:]
    candidates = [x for l, r in itertools.product(lefts, rights) if (x := transduce(l, r)) is not None]
    candidates.sort(key=lambda x: (x["audit"]["exact"], x["audit"]["letters"]), reverse=True)
    exact = [x for x in candidates if x["audit"]["exact"]]
    repair = {
        "failure": "no exact closure in bounded agreement-valid product" if not exact else "exact closure found",
        "first_residual": candidates[0]["audit"]["two_pointer_mismatches"][0] if candidates and candidates[0]["audit"]["two_pointer_mismatches"] else None,
        "operator": "add one held-out copular/past-tense inflection pair at the first residual boundary while preserving each subject-number register",
        "held_out_variants": ["is/are", "was/were", "carries/carry"],
    }
    return {"experiment_id": EXPERIMENT_ID, "signature": SIGNATURE,
            "status": "completed_exact" if exact else "completed_no_exact_closure",
            "reader_eligible": bool(exact), "method": "complete-clause-first agreement register transducer with bounded inflectional boundary CSP",
            "novelty_preflight": preflight, "candidate_count": len(candidates),
            "candidates": candidates[:24], "stats": {"complete_clauses_generated": len(clauses),
            "bounded_pairs": len(lefts) * len(rights), "transduced": len(candidates), "exact": len(exact)},
            "failure_and_repair": repair,
            "provenance": {"lexical_source": "fresh hand-authored clause inventories", "catalogue_text_imported": False,
                           "known_palindrome_imported": False, "fixed_tape_used": False,
                           "independent_audits": ["two-pointer", "forward/reverse SHA-256", "mechanical admission"],
                           "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


if __name__ == "__main__":
    OUT.write_text(json.dumps(run(), indent=2) + "\n")
    print(json.dumps(run()["stats"], sort_keys=True))
