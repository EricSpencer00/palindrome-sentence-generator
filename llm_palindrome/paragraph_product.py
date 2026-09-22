"""Exact dual-grammar products with independently placed sentence seams.

The useful paragraph construction is not a list of already-palindromic
sentences.  A left half parses as ``A, B, ...`` while the reverse-facing right
half parses independently as ``..., B-prime, A-prime``.  Sentence boundaries
are part of the two parses and are deliberately allowed to occur at different
letter offsets.  A reflected sentence can therefore cut across a sentence
boundary on the other side even though the complete paragraph is exact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import accumulate
from typing import Callable, Sequence

from .dual_parse import letter_tape, word_residual_search


Slot = tuple[str, tuple[str, ...]]
SentencePlan = tuple[Slot, ...]


def _internal_boundaries(sentences: Sequence[str]) -> tuple[int, ...]:
    lengths = [len(letter_tape(sentence)) for sentence in sentences]
    return tuple(accumulate(lengths[:-1]))


def _proper_palindromic_sentence_spans(sentences: Sequence[str]) -> list[dict]:
    """Return proper contiguous sentence spans that close independently."""
    out: list[dict] = []
    for start in range(len(sentences)):
        for stop in range(start + 1, len(sentences) + 1):
            if start == 0 and stop == len(sentences):
                continue
            tape = letter_tape(" ".join(sentences[start:stop]))
            if tape and tape == tape[::-1]:
                out.append({"start": start, "stop": stop, "letters": len(tape)})
    return out


def audit_staggered_abba(left_sentences: Sequence[str],
                         right_sentences: Sequence[str]) -> dict:
    """Audit an exact paragraph whose two parses have independent seams.

    ``right_sentences`` are supplied in normal reading order.  Their boundary
    offsets are reflected into the left-to-right half-tape coordinate system,
    so equality with a left offset identifies a forbidden preclosed block.
    """
    left_sentences = tuple(left_sentences)
    right_sentences = tuple(right_sentences)
    sentences = left_sentences + right_sentences
    left = letter_tape(" ".join(left_sentences))
    right = letter_tape(" ".join(right_sentences))
    full = left + right
    left_boundaries = _internal_boundaries(left_sentences)
    right_reading_boundaries = _internal_boundaries(right_sentences)
    reflected_right_boundaries = tuple(
        sorted(len(right) - offset for offset in right_reading_boundaries)
    )
    aligned = sorted(set(left_boundaries) & set(reflected_right_boundaries))
    whole_sentence_mirrors = []
    for li, left_sentence in enumerate(left_sentences):
        lt = letter_tape(left_sentence)
        for ri, right_sentence in enumerate(right_sentences):
            rt = letter_tape(right_sentence)
            if lt and lt == rt[::-1]:
                whole_sentence_mirrors.append({"left": li, "right": ri,
                                               "letters": len(lt)})
    proper_spans = _proper_palindromic_sentence_spans(sentences)
    forward_hash = hashlib.sha256(full.encode()).hexdigest()
    reverse_hash = hashlib.sha256(full[::-1].encode()).hexdigest()
    half_equation = bool(left) and left == right[::-1]
    staggered = bool(left_boundaries and reflected_right_boundaries) and not aligned
    return {
        "letters": len(full),
        "left_letters": len(left),
        "right_letters": len(right),
        "half_equation": half_equation,
        "two_pointer_exact": bool(full) and full == full[::-1],
        "sha256_forward": forward_hash,
        "sha256_reverse": reverse_hash,
        "sha_equal": forward_hash == reverse_hash,
        "left_boundaries": list(left_boundaries),
        "reflected_right_boundaries": list(reflected_right_boundaries),
        "aligned_internal_boundaries": aligned,
        "sentence_boundaries_staggered": staggered,
        "whole_sentence_mirrors": whole_sentence_mirrors,
        "proper_palindromic_sentence_spans": proper_spans,
        "cross_sentence_coupled": (
            half_equation and staggered and not whole_sentence_mirrors
            and not proper_spans
        ),
    }


def _flatten(plans: Sequence[SentencePlan], side: str) -> tuple[Slot, ...]:
    return tuple(
        (f"{side}{sentence_index}:{role}", alternatives)
        for sentence_index, sentence in enumerate(plans)
        for role, alternatives in sentence
    )


def _split(words: Sequence[str], plans: Sequence[SentencePlan]) -> tuple[str, ...]:
    sentences = []
    offset = 0
    for plan in plans:
        width = len(plan)
        sentence = " ".join(words[offset:offset + width])
        sentences.append(sentence[:1].upper() + sentence[1:] + ".")
        offset += width
    if offset != len(words):
        raise AssertionError("word path does not match paragraph plan")
    return tuple(sentences)


def staggered_abba_search(
    left_plans: Sequence[SentencePlan],
    right_plans: Sequence[SentencePlan],
    *,
    max_states: int = 250_000,
    max_results: int = 100,
    allow_partial: Callable[[tuple[str, ...], tuple[str, ...]], bool] | None = None,
) -> dict:
    """Intersect paragraph grammars and retain only cross-sentence closures."""
    raw = word_residual_search(
        _flatten(left_plans, "L"),
        _flatten(right_plans, "R"),
        max_states=max_states,
        max_results=max_results,
        allow_partial=allow_partial,
        # Closure at a word edge is representation-dependent.  The paragraph
        # shortcut test is instead made on complete sentence spans below.
        reject_intermediate_closure=False,
    )
    candidates = []
    for row in raw["results"]:
        left = _split(tuple(row["left"].split()), left_plans)
        right = _split(tuple(row["right"].split()), right_plans)
        audit = audit_staggered_abba(left, right)
        candidates.append({
            "left_sentences": list(left),
            "right_sentences": list(right),
            "rendered": " ".join(left + right),
            "audit": audit,
            "cross_sentence_coupled": audit["cross_sentence_coupled"],
        })
    return {
        "candidates": candidates,
        "cross_sentence_candidates": [
            row for row in candidates if row["cross_sentence_coupled"]
        ],
        "states": raw["states"],
        "transitions": raw["transitions"],
        "cap_reached": raw["cap_reached"],
        "dead_frontiers": raw["dead_frontiers"],
    }
