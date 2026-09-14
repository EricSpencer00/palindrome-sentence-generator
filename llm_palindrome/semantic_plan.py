"""Small, auditable semantic-plan object for plan-conditioned exact search.

Plans are ranking context, not a readability certificate.  The exact lattice
still owns all letter transitions, and a separate evaluator would be required
for entailment or human readability.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, Iterable, Sequence


def _word_set(words: Iterable[str]) -> frozenset[str]:
    tokens: set[str] = set()
    for word in words:
        if isinstance(word, str):
            tokens.update(re.findall(r"[a-z]+", word.casefold()))
    return frozenset(tokens)


@dataclass(frozen=True)
class SemanticFact:
    fact_id: str
    subject: str
    predicate: str
    object: str
    terms: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticFact":
        if not isinstance(value, dict) or set(value) != {"fact_id", "subject", "predicate", "object", "terms"}:
            raise ValueError("semantic_fact_schema_invalid")
        if not all(isinstance(value[key], str) and value[key].strip() for key in ("fact_id", "subject", "predicate", "object")):
            raise ValueError("semantic_fact_fields_invalid")
        terms = value["terms"]
        if not isinstance(terms, list) or not terms or not all(isinstance(term, str) and re.fullmatch(r"[a-z]+(?: [a-z]+)*", term.casefold()) for term in terms):
            raise ValueError("semantic_fact_terms_invalid")
        return cls(value["fact_id"], value["subject"], value["predicate"], value["object"], tuple(term.casefold() for term in terms))

    def as_dict(self) -> dict[str, Any]:
        return {"fact_id": self.fact_id, "subject": self.subject, "predicate": self.predicate, "object": self.object, "terms": list(self.terms)}


@dataclass(frozen=True)
class SemanticPlan:
    intent: str
    facts: tuple[SemanticFact, ...]
    links: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SemanticPlan":
        if not isinstance(value, dict) or set(value) != {"intent", "facts", "links"}:
            raise ValueError("semantic_plan_schema_invalid")
        if not isinstance(value["intent"], str) or not value["intent"].strip() or not isinstance(value["facts"], list) or not isinstance(value["links"], list) or not all(isinstance(link, str) for link in value["links"]):
            raise ValueError("semantic_plan_fields_invalid")
        facts = tuple(SemanticFact.from_dict(fact) for fact in value["facts"])
        if not 2 <= len(facts) <= 4 or len({fact.fact_id for fact in facts}) != len(facts):
            raise ValueError("semantic_plan_requires_two_to_four_distinct_facts")
        return cls(value["intent"], facts, tuple(value["links"]))

    def as_dict(self) -> dict[str, Any]:
        return {"intent": self.intent, "facts": [fact.as_dict() for fact in self.facts], "links": list(self.links)}

    @property
    def sha256(self) -> str:
        return sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def covered_facts(self, words: Sequence[str]) -> tuple[str, ...]:
        observed = _word_set(words)
        return tuple(fact.fact_id for fact in self.facts if _word_set(fact.terms) <= observed)

    def coverage_vector(self, words: Sequence[str]) -> tuple[bool, ...]:
        covered = set(self.covered_facts(words))
        return tuple(fact.fact_id in covered for fact in self.facts)

    def minimum_coverage(self, words: Sequence[str]) -> int:
        return sum(self.coverage_vector(words))


def render_plan_prompt() -> str:
    """Frozen request for plan context; it never asks the model for a palindrome."""
    payload = {
        "task": "Write one original ordinary-English sentence and its compact predicate plan.",
        "hard_rules": [
            "Return one JSON object and nothing else.",
            "The sentence must be a complete concrete thought with 100 to 160 ASCII letters.",
            "Use two to four facts with stable entity IDs, a predicate, an object, and exact lower-case content terms.",
            "Do not mention special text constraints, wordplay, or the writing process.",
        ],
        "response_schema": {
            "draft": "complete ordinary English sentence",
            "plan": {"intent": "one-sentence proposition", "facts": [{"fact_id": "f1", "subject": "e1", "predicate": "...", "object": "e2", "terms": ["...", "..."]}], "links": ["f1 causes f2"]},
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
