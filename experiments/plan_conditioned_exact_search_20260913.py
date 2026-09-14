"""Plan-conditioned exact lattice search.

The existing word lattice is the only generator: every child is an exact
letter-palindrome consequence of a legal overhang transition.  A frozen
semantic plan changes ranking and records fact coverage; it never supplies
letters, word-order mirrors, or a readability claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters
from llm_palindrome.exact_editor import new_state, surface_audit
from llm_palindrome.generate import ZipfScorer, build_vocab
from llm_palindrome.search import WordTries, beam_search
from llm_palindrome.semantic_plan import SemanticPlan
from llm_palindrome.textify import textify


MIN_LETTERS = 100
MAX_LETTERS = 160


class PlanConditionedScorer:
    """Blend local language ordering with frozen fact-coverage gains."""

    def __init__(self, plan: SemanticPlan, base: Any, fact_weight: float = 12.0):
        self.plan = plan
        self.base = base
        self.fact_weight = fact_weight

    def word_delta(self, left, right, placement, word, growth):
        # ``beam_search`` passes the child frontiers here.  Remove the newly
        # added unit before measuring the gain; comparing the child with itself
        # made every fact bonus identically zero in the first pilot.
        if placement == "L":
            before_words = tuple(left[:-1]) + tuple(right)
        elif placement == "R":
            before_words = tuple(left) + tuple(right[1:])
        else:
            raise ValueError(f"unknown placement: {placement!r}")
        before = self.plan.minimum_coverage(before_words)
        after = self.plan.minimum_coverage(tuple(left) + tuple(right))
        return self.base.word_delta(left, right, placement, word, growth) + self.fact_weight * (after - before)


def plan_candidate_audit(text: str, plan: SemanticPlan, *, seed: int) -> dict[str, Any]:
    tape = normalize_letters(text)
    midpoint = len(tape) // 2
    half = tape[:midpoint]
    center = tape[midpoint] if len(tape) % 2 else ""
    editor_state = new_state(half_text=half, center_text=center, intent=plan.intent, surface_hint=text)
    audit = surface_audit(editor_state, text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS)
    words = audit["words"]
    return {
        "seed": seed,
        "rendered": text,
        "render_sha256": sha256(text.encode()).hexdigest(),
        "letters": len(tape),
        "plan_sha256": plan.sha256,
        "plan_fact_coverage": {fact.fact_id: fact.fact_id in plan.covered_facts(words) for fact in plan.facts},
        "plan_minimum_coverage": plan.minimum_coverage(words),
        "exact_editor_audit": audit,
        "mechanical_checks": mechanical_admission_checks(text, min_letters=MIN_LETTERS, max_letters=MAX_LETTERS),
        "human_reader_study": "not_run",
    }


def run(plan: SemanticPlan, *, seeds: int = 32, vocabulary_size: int = 18000, beam: int = 240, fact_weight: float = 12.0) -> dict[str, Any]:
    vocabulary = build_vocab(vocabulary_size)
    tries = WordTries(vocabulary)
    scorer = PlanConditionedScorer(plan, ZipfScorer(), fact_weight=fact_weight)
    records: list[dict[str, Any]] = []
    for seed in range(seeds):
        words = beam_search(tries, scorer, min_letters=MIN_LETTERS, beam_width=beam, max_steps=220,
                            candidate_limit=1200, seed=seed, diversity=1.3)
        if not words:
            continue
        text = textify(words)
        record = plan_candidate_audit(text, plan, seed=seed)
        record["words"] = words
        record["mechanically_eligible"] = all(record["mechanical_checks"].values())
        records.append(record)
    eligible = [record for record in records if record["mechanically_eligible"]]
    return {
        "status": "plan_conditioned_exact_search_complete",
        "config": {"seeds": seeds, "vocabulary_size": len(vocabulary), "beam": beam, "fact_weight": fact_weight,
                    "candidate_letter_range": [MIN_LETTERS, MAX_LETTERS], "exactness_owner": "WordTries/beam_search plus independent exact_editor audit",
                    "machine_readability_certification": False},
        "plan": plan.as_dict(),
        "plan_sha256": plan.sha256,
        "records": records,
        "mechanically_eligible": eligible,
        "reader_gate": "No readability claim; eligible surfaces require blinded human readers with intact prose and shuffled controls.",
        "provenance": {"generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "lexicon_path": "data/lexicon.txt"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path, help="frozen SemanticPlan JSON")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--vocabulary-size", type=int, default=18000)
    parser.add_argument("--beam", type=int, default=240)
    parser.add_argument("--fact-weight", type=float, default=12.0)
    args = parser.parse_args()
    plan = SemanticPlan.from_dict(json.loads(args.plan.read_text()))
    result = run(plan, seeds=args.seeds, vocabulary_size=args.vocabulary_size, beam=args.beam, fact_weight=args.fact_weight)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(result["records"]), "mechanically_eligible": len(result["mechanically_eligible"])}, sort_keys=True))


if __name__ == "__main__":
    main()
