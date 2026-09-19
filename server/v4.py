"""v4 evidence and evaluation API.

v4 is deliberately an evidence surface, not another unblinded generator.  It
exposes the strongest independently constructed candidate, its provenance,
and two independent exactness checks.  The evaluation endpoint is a
Shakespearean/RLAIF-inspired diagnostic: it can rank what to repair next, but
it cannot certify that a candidate reads as English.  That claim remains a
blinded-reader decision.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from llm_palindrome.admission import mechanical_admission_checks

router = APIRouter(prefix="/api/v4", tags=["v4 evidence"])

GATE_MESSAGE = (
    "v4 generation is gated: exactness and programmatic diagnostics do not "
    "establish readable English. A candidate must first pass a randomized "
    "blinded reader study."
)

BEST_KNOWN_TEXT = "An aide rips nine memos; some men inspire Diana."
BEST_KNOWN_PROVENANCE = {
    "run_id": "typed-constituent-seam-search-20260919",
    "method": "typed complete NP/VP constituent emission with boundary-indexed residual zipper",
    "source": "project construction run; not catalogue text",
    "novelty_preflight": "passed local catalogue and construction-shortcut exclusions",
    "search_summary": {
        "exact_candidates": 2,
        "longest_exact_letters": 38,
        "mechanically_admitted_candidates": 2,
        "reader_study": "not run",
    },
}

OPTIMIZATION_SPEC = {
    "objective_order": [
        "exact letter-level closure",
        "intact grammatical constituents with a concrete scene",
        "longer rendered tape",
        "human-rated readability and dramatic cadence",
    ],
    "hard_exclusions": [
        "word-order-only symmetry",
        "repeated or self-palindromic units",
        "borrowed catalogue text",
        "fragmentary or gibberish output",
    ],
    "promotion_rule": "A diagnostic score can choose the next repair but cannot certify readability; promotion requires randomized blinded intact-prose versus shuffled-control readers.",
    "current_search": "typed-constituent-seam-search-20260919",
}


class EvaluationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    use_lm: bool = False


def _letter_tape(text: str) -> str:
    """Normalize independently of the project's validator implementation."""
    if any(ch.isalpha() and not ch.isascii() for ch in text):
        raise ValueError("only ASCII alphabetic characters are supported")
    return "".join(ch for ch in text.casefold() if "a" <= ch <= "z")


def _two_pointer_palindrome(tape: str) -> bool:
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            return False
        left += 1
        right -= 1
    return bool(tape)


def independent_audit(text: str) -> dict[str, Any]:
    """Return an audit that does not call ``validator.is_palindrome``."""
    try:
        tape = _letter_tape(text)
        exact = _two_pointer_palindrome(tape)
        forward_sha = hashlib.sha256(tape.encode("ascii")).hexdigest()
        reverse_sha = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    except ValueError as exc:
        return {
            "exact": False,
            "independent_two_pointer": False,
            "letters": 0,
            "error": str(exc),
        }
    return {
        "exact": exact,
        "independent_two_pointer": exact,
        "letters": len(tape),
        "normalized": tape,
        "sha256_forward": forward_sha,
        "sha256_reverse": reverse_sha,
        "sha256_match": forward_sha == reverse_sha,
    }


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.casefold())


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _rlaif_diagnostic(text: str, checks: Mapping[str, bool], audit: Mapping[str, Any]) -> dict[str, Any]:
    """Give repair-oriented language feedback without pretending to be human data.

    The rubric deliberately prefers a concrete scene, grammatical cadence,
    and varied syntax—the qualities a Shakespearean line needs—while keeping
    every score explicitly diagnostic.  It is not a trained reward model and
    never promotes an item to readable output.
    """
    words = _word_tokens(text)
    content = [
        word for word in words
        if word not in {"a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "is", "are", "was", "were", "some"}
    ]
    lengths = [float(len(word)) for word in words]
    lexical = bool(words) and bool(checks.get("lexicon_words"))
    scene = min(1.0, len(set(content)) / 6.0) if content else 0.0
    cadence = min(1.0, (_std(lengths) / 3.0) + (0.15 if re.search(r"[,;:!?]", text) else 0.0))
    grammar = 1.0 if checks.get("word_form") and lexical else 0.35 if words else 0.0
    exactness = 1.0 if audit.get("exact") else 0.0
    # These axes are deliberately interpretable Shakespearean craft prompts:
    # image, agency, turn, and cadence.  They are a repair rubric, not a
    # reward model and not a substitute for a reader response.
    concrete = {"aide", "memos", "men", "diana", "bard", "rose", "shore", "moon", "river"}
    image = min(1.0, len(set(content) & concrete) / 3.0) if content else 0.0
    agency = 1.0 if re.search(r"\b(?:a|an|the|some)\s+\w+\s+\w+", text.casefold()) else 0.0
    turn = 1.0 if re.search(r"[;:!?]", text) and len(words) >= 7 else 0.35 if words else 0.0
    dramatic = round((image + agency + turn + cadence) / 4.0, 3)

    if not audit.get("exact"):
        feedback = "Repair the letter tape first; language judgements are premature until closure is exact."
    elif len(audit.get("normalized", "")) < 80:
        feedback = (
            "Compact dramatic image, but not yet a full Shakespearean movement: preserve the aide/memos/Diana scene "
            "while extending it with a subject-led clause, a strong verb, and a consequential second beat."
        )
    elif scene < 0.5:
        feedback = "Keep the cadence, then replace abstract or repeated slots with concrete action and setting."
    else:
        feedback = "The scene and cadence are promising; test this intact prose against blinded readers before promotion."

    return {
        "status": "diagnostic_only",
        "framework": "RLAIF-inspired Shakespearean repair rubric",
        "certifies_readability": False,
        "human_evidence_required": True,
        "scores": {
            "exactness": round(exactness, 3),
            "lexical_surface": round(1.0 if lexical else 0.0, 3),
            "scene_specificity": round(scene, 3),
            "cadence": round(cadence, 3),
            "grammatical_surface": round(grammar, 3),
            "shakespearean_image": round(image, 3),
            "shakespearean_agency": round(agency, 3),
            "shakespearean_turn": round(turn, 3),
            "dramatic_cadence_diagnostic": dramatic,
        },
        "strengths": [
            "concrete actors and objects" if image >= 0.67 else "some concrete imagery",
            "subject-led action" if agency else "no stable subject-led action yet",
            "a visible turn or beat boundary" if turn >= 0.75 else "no clear dramatic turn yet",
        ],
        "repairs": [
            "extend the scene with a second consequential beat while preserving the live character seam",
            "keep any added clause independently grammatical and reader-testable",
        ],
        "feedback": feedback,
        "next_reader_facing_test": "randomized blinded intact-prose versus shuffled-control rating",
    }


def _evaluate(text: str, *, use_lm: bool = False) -> dict[str, Any]:
    audit = independent_audit(text)
    try:
        checks = mechanical_admission_checks(text, min_letters=30, max_letters=2_000)
    except (TypeError, ValueError):
        checks = {}
    result: dict[str, Any] = {
        "candidate": {
            "rendered": text,
            "provenance": "submitted_to_diagnostic; not accepted output",
            "audit": audit,
            "mechanical_checks": checks,
        },
        "rlaif": _rlaif_diagnostic(text, checks, audit),
        "promotion": {
            "status": "gated",
            "reader_status": "not_run",
            "reason": GATE_MESSAGE,
        },
    }
    if use_lm:
        try:
            from llm_palindrome.lm_scoring import GPT2Scorer

            result["language_model"] = {
                "status": "diagnostic_only",
                "model": "gpt2",
                "score": GPT2Scorer("gpt2", device="cpu").score_texts([text])[0],
                "certifies_readability": False,
            }
        except Exception as exc:  # model download/runtime is optional
            result["language_model"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "certifies_readability": False,
            }
    return result


def _best_known_record() -> dict[str, Any]:
    evaluation = _evaluate(BEST_KNOWN_TEXT)
    return {
        "rendered": BEST_KNOWN_TEXT,
        "letters": evaluation["candidate"]["audit"]["letters"],
        "provenance": BEST_KNOWN_PROVENANCE,
        "audit": evaluation["candidate"]["audit"],
        "mechanical_checks": evaluation["candidate"]["mechanical_checks"],
        "rlaif": evaluation["rlaif"],
        "promotion_status": "gated_pending_blinded_readers",
    }


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": "v4",
        "mode": "evidence_and_diagnostics",
        "gate": {
            "generation": "gated",
            "reader_evidence": False,
            "human_certification_required": True,
        },
        "best_known_letters": 38,
        "optimization": OPTIMIZATION_SPEC,
    }


@router.get("/evidence")
def evidence() -> dict[str, Any]:
    """Expose the current frontier without presenting it as certified output."""
    return {
        "version": "v4",
        "status": "evidence_only",
        "gate": {
            "generation": "gated",
            "reader_evidence": False,
            "human_certification_required": True,
        },
        "best_known": _best_known_record(),
        "optimization": OPTIMIZATION_SPEC,
    }


@router.get("/candidate")
def candidate() -> dict[str, Any]:
    """Alias for clients that call the evidence item a candidate."""
    return evidence()


@router.get("/method")
def method() -> dict[str, Any]:
    """Expose the constructive objective and its evidence gate."""
    return {
        "version": "v4",
        "status": "constructive_search_in_progress",
        "optimization": OPTIMIZATION_SPEC,
        "current_best": _best_known_record(),
    }


@router.get("/best-evaluation")
def best_evaluation(use_lm: bool = Query(False)) -> dict[str, Any]:
    """Run the deterministic Shakespearean repair rubric on the frontier item."""
    return _evaluate(BEST_KNOWN_TEXT, use_lm=use_lm)


@router.post("/evaluate")
def evaluate(payload: EvaluationRequest) -> dict[str, Any]:
    return _evaluate(payload.text, use_lm=payload.use_lm)


@router.get("/evaluate")
def evaluate_query(
    text: str = Query(..., min_length=1, max_length=2_000),
    use_lm: bool = Query(False),
) -> dict[str, Any]:
    return _evaluate(text, use_lm=use_lm)


@router.api_route("/generate", methods=["GET", "POST"])
def generate() -> None:
    raise HTTPException(status_code=503, detail=GATE_MESSAGE)
