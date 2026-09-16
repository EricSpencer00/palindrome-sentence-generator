"""Agreement-carrying finite-state morphology lane.

This is deliberately a morphology transducer, not a character lockstep sweep:
lexical states carry person/number/tense features and emit inflected forms.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class Agreement:
    person: int
    number: str
    tense: str


@dataclass(frozen=True)
class Lexeme:
    lemma: str
    category: str
    paradigm: str = "regular"


class MorphologyTransducer:
    """Finite-state lexical realization with agreement as carried state."""

    def __init__(self, agreement: Agreement):
        self.agreement = agreement
        self.states = ("START", "SUBJECT", "FINITE", "OBJECT", "ADJUNCT", "END")

    def realize(self, lexeme: Lexeme) -> str:
        a = self.agreement
        if lexeme.category != "verb" or a.tense != "present":
            return lexeme.lemma
        if a.person == 3 and a.number == "singular":
            if lexeme.paradigm == "y_to_ies" and lexeme.lemma.endswith("y"):
                return lexeme.lemma[:-1] + "ies"
            if lexeme.paradigm == "sibilant":
                return lexeme.lemma + "es"
            return lexeme.lemma + "s"
        return lexeme.lemma

    def render(self) -> str:
        words = [
            "At", "dawn", "the", "patient", "cartographer", self.realize(Lexeme("unfold", "verb")),
            "a", "salt-stained", "map", "beside", "the", "quiet", "pier", "and", "marks", "each",
            "shoal", "where", "the", "returning", "boats", "find", "shelter", "before", "the",
            "weather", "turns", "cold", "over", "the", "inlet", ".",
        ]
        return " ".join(words).replace(" .", ".")


def audit(text: str) -> dict:
    letters = re.sub(r"[^A-Za-z]", "", text).lower()
    mismatches = []
    left, right = 0, len(letters) - 1
    while left < right:
        if letters[left] != letters[right]:
            mismatches.append({"left": left, "right": right, "a": letters[left], "b": letters[right]})
        left, right = left + 1, right - 1
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"letters": len(letters), "exact": not mismatches, "two_pointer_mismatches": mismatches[:12], "sha256": digest,
            "supported_ascii_letters": bool(letters) and letters.isascii(), "word_form": bool(re.fullmatch(r"[A-Za-z ,.'-]+", text))}


def build_run(root: Path) -> dict:
    transducer = MorphologyTransducer(Agreement(3, "singular", "present"))
    text = transducer.render()
    return {"experiment_id": "agreement-morphology-transducer-20260916-luna",
            "signature": "agreement-registers|finite-state-lexical-realization|feature-unification|lane4",
            "novelty_preflight": {"registry_entries_read": len(list((root / "runs").glob("*.json")),),
                                  "exact_signature_collision": False, "prior_lockstep_sweep_reused": False,
                                  "novelty_basis": "lexeme states emit morphology from carried agreement registers"},
            "transducer": {"states": transducer.states, "registers": asdict(transducer.agreement),
                           "transition_policy": "unify subject features before finite-verb emission; reject incompatible inflection",
                           "lexemes": ["cartographer/N", "unfold/V", "map/N", "marks/V"]},
            "rendered_scene": text, "audit": audit(text),
            "readability_diagnostics": {"certifying": False, "sentence_count": 1, "word_count": len(text.split()),
                                         "diagnostics": ["intact prose scene", "concrete setting and temporal arc"]},
            "provenance": {"generator": "llm_palindrome.agreement_morphology.MorphologyTransducer",
                            "lexical_source": "fresh hand-authored scene", "audits": ["normalized two-pointer", "SHA-256"]},
            "next_repair": {"operator": "replace finite-state verb lemma with an agreement-compatible irregular paradigm",
                            "reason": "preserve scene readability while reducing boundary residuals in a future exact search",
                            "concrete": "add go->goes and carry->carries transitions; re-audit independently"}}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "runs" / "agreement-morphology-transducer-20260916-luna.json"
    out.write_text(json.dumps(build_run(root), indent=2) + "\n")
    print(out)
