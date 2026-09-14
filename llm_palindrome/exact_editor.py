"""Exact-palindromic character editor with independent surface resegmentation.

The editor's state is an exact character tape from its first construction step:
``half + center + reverse(half)``.  A proposer edits only the half tape; the
other characters are mathematical consequences, never a supplied word-order
mirror.  A proposed rendered surface is accepted only when its own letters
equal that tape.  Word boundaries, punctuation, and syntactic attachments are
therefore free to cross the reflection seam.

This module deliberately does not certify readability and does not turn a
lexical segmentation into a paper example.  The shared admission gate and a
human reader study remain downstream.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import re
from typing import Any, Iterable

from .admission import mechanical_admission_checks, normalize_letters, tokenize


WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _digest(value: Any) -> str:
    import json

    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _ascii_tape(text: str) -> str:
    """Independent letter normalizer for the editor's internal invariant."""
    if not isinstance(text, str):
        raise ValueError("text_must_be_string")
    if any(char.isalpha() and not char.isascii() for char in text):
        raise ValueError("unsupported_non_ascii_alphabetic_character")
    return "".join(char.lower() for char in text if char.isascii() and char.isalpha())


def _require_palindromic_center(center: str) -> str:
    tape = _ascii_tape(center)
    if tape != tape[::-1]:
        raise ValueError("center_must_be_palindromic")
    return tape


@dataclass(frozen=True)
class HalfSpan:
    start: int
    end: int


@dataclass(frozen=True)
class ExactEditorState:
    state_id: str
    half_tape: str
    center_tape: str
    intent: str
    surface_hint: str
    parent_id: str | None = None
    edit_history: tuple[dict[str, Any], ...] = ()

    @property
    def full_tape(self) -> str:
        return self.half_tape + self.center_tape + self.half_tape[::-1]

    @property
    def letters(self) -> int:
        return len(self.full_tape)


def new_state(*, half_text: str, center_text: str = "", intent: str = "", surface_hint: str = "") -> ExactEditorState:
    half = _ascii_tape(half_text)
    center = _require_palindromic_center(center_text)
    if not half and not center:
        raise ValueError("empty_exact_tape")
    payload = {"parent": None, "half": half, "center": center, "intent": intent, "surface_hint": surface_hint, "history": []}
    return ExactEditorState(_digest(payload), half, center, intent, surface_hint, None, ())


def paired_span_edit(state: ExactEditorState, span: HalfSpan, replacement: str, *, source: dict[str, str], notes: str = "") -> tuple[ExactEditorState | None, dict[str, Any]]:
    """Replace a half-tape span and reflect it mathematically in the exact tape."""
    event: dict[str, Any] = {
        "operation": "paired_half_span_edit",
        "parent_state_id": state.state_id,
        "source": source,
        "input": {"span": asdict(span), "replacement": replacement, "notes": notes},
        "accepted": False,
    }
    try:
        if not (0 <= span.start <= span.end <= len(state.half_tape)):
            raise ValueError("half_span_out_of_bounds")
        edited = _ascii_tape(replacement)
        if not edited:
            raise ValueError("replacement_must_contain_letters")
        next_half = state.half_tape[:span.start] + edited + state.half_tape[span.end:]
        history = state.edit_history + ({"operation": "paired_half_span_edit", "span": asdict(span), "replacement": edited, "source": source, "notes": notes},)
        child = ExactEditorState(
            _digest({"parent": state.state_id, "half": next_half, "center": state.center_tape, "intent": state.intent, "surface_hint": state.surface_hint, "history": history}),
            next_half,
            state.center_tape,
            state.intent,
            state.surface_hint,
            state.state_id,
            history,
        )
        event.update({"accepted": True, "child_state_id": child.state_id, "half_letters": len(next_half), "full_letters": child.letters, "full_tape": child.full_tape, "exact_by_construction": child.full_tape == child.full_tape[::-1]})
        return child, event
    except (TypeError, ValueError) as error:
        event["rejection"] = str(error)
        return None, event


def lexical_surface_evidence(state: ExactEditorState, *, max_segmentations: int = 512) -> dict[str, Any]:
    """Record independent word-boundary evidence without promoting any surface."""
    # Importing the pilot's broad lexical chart keeps this editor's first
    # milestone aligned with the existing frozen lexicon, while exactness is
    # still checked independently here and in `surface_audit` below.
    from experiments.assisted_candidate_construction_pilot_20260913 import analyze_tape, enumerate_segmentations

    analysis = analyze_tape(state.full_tape)
    segmentations, truncated = enumerate_segmentations(state.full_tape, limit=max_segmentations)
    return {
        "tape": state.full_tape,
        "letters": state.letters,
        "complete_segmentation_count": analysis.complete_segmentations,
        "boundary_positions": analysis.boundary_positions,
        "open_prefixes": analysis.open_prefixes,
        "materialized_segmentations": [list(words) for words in segmentations],
        "materialized_count": len(segmentations),
        "truncated": truncated,
        "surface_rendering_is_not_certification": True,
    }


def materialized_surface_audits(
    state: ExactEditorState, *, max_segmentations: int = 512,
    min_letters: int = 0, max_letters: int = 100000,
) -> list[dict[str, Any]]:
    """Audit every materialized lexical rendering of an exact tape.

    The editor owns the tape but never chooses a preferred segmentation.  This
    helper makes each host-generated rendering explicit, independently checks
    its letters, and retains only surfaces that pass the shared mechanical
    gate.  The returned list is construction evidence, not a readability
    judgment; a human study is still required before promotion.
    """
    from .textify import textify

    evidence = lexical_surface_evidence(state, max_segmentations=max_segmentations)
    if evidence["truncated"]:
        return []
    audited = []
    for words in evidence["materialized_segmentations"]:
        rendered = textify(words)
        audit = surface_audit(state, rendered, min_letters=min_letters,
                              max_letters=max_letters)
        if audit["mechanically_eligible"]:
            audited.append(audit)
    return audited


def surface_audit(state: ExactEditorState, rendered: str, *, min_letters: int = 0, max_letters: int = 100000) -> dict[str, Any]:
    """Audit a supplied surface; the host never invents its word boundaries."""
    editor_tape = _ascii_tape(rendered)
    shared_tape = normalize_letters(rendered)
    central = mechanical_admission_checks(rendered, min_letters=min_letters, max_letters=max_letters)
    words = tokenize(rendered)
    independent = {
        "normalizer": "exact_editor_explicit_ascii_scan_v1",
        "tape": editor_tape,
        "letter_count": len(editor_tape),
        "direct_symmetric_position_comparison": all(editor_tape[index] == editor_tape[-1 - index] for index in range(len(editor_tape))),
        "matches_state_tape": editor_tape == state.full_tape,
        "central_normalizer_agrees": editor_tape == shared_tape,
    }
    return {
        "rendered": rendered,
        "render_sha256": sha256(rendered.encode()).hexdigest(),
        "words": list(words),
        "state_id": state.state_id,
        "intent": state.intent,
        "letters": len(editor_tape),
        "central_mechanical_checks": central,
        "independent_exactness": independent,
        "mechanically_eligible": all(central.values()) and all(independent.values()),
        "provenance": {"state_id": state.state_id, "edit_history": list(state.edit_history)},
        "human_reader_study": "not_run",
    }


def replay_edits(initial: ExactEditorState, edits: Iterable[tuple[HalfSpan, str, dict[str, str], str]]) -> tuple[ExactEditorState, list[dict[str, Any]]]:
    """Replay a deterministic edit trace and retain every acceptance/rejection."""
    state = initial
    events: list[dict[str, Any]] = []
    for span, replacement, source, notes in edits:
        child, event = paired_span_edit(state, span, replacement, source=source, notes=notes)
        events.append(event)
        if child is not None:
            state = child
    return state, events
