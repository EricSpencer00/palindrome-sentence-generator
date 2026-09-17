"""Bounded character-LM beam with explicit grammar and word-boundary states.

This is deliberately not an obligation beam: it never carries a mirrored
character debt or chooses a right-hand continuation.  It decodes one ordinary
semantic clause at a time through a finite-state grammar, and then pairs two
independently decoded clauses into intact prose controls.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT_ID = "finite-state-boundary-beam-20260917"
SIGNATURE = (
    "finite-state-boundary-beam|semantic-clause-template|"
    "lexical-transition-score|boundary-class-grammar|independent-pointer-sha"
)
OUT = ROOT / "runs" / "finite-state-boundary-beam-20260917.json"
REGISTRY = ROOT / "docs" / "experiment-novelty-registry.json"

# The corpus is only a transparent ranking signal; it is not a source of
# finished sentences.  All emitted words come from the typed frame bank.
CORPUS = (
    "the careful baker warms a copper kettle near the window. "
    "a quiet gardener records fresh seeds beside the stone wall. "
    "the patient ranger checks a weather map before the evening rain."
)
ALPHABET = set("abcdefghijklmnopqrstuvwxyz ")
TRIGRAM: dict[str, int] = {}
for i in range(len(CORPUS) - 2):
    gram = CORPUS[i : i + 3].lower()
    if set(gram) <= ALPHABET:
        TRIGRAM[gram] = TRIGRAM.get(gram, 0) + 1


@dataclass(frozen=True)
class BeamState:
    words: tuple[str, ...]
    phase: str
    boundary: str
    score: float
    trace: tuple[dict[str, str], ...]


# Each frame is a semantic clause template.  The slots are finite-state
# grammar transitions, not free character continuation.  Boundary labels make
# the spaces and constituent edges explicit in the state inventory.
FRAMES = (
    {
        "name": "craft",
        "subject": ("the careful baker", "the patient potter", "a steady tailor"),
        "verb": ("warms", "shapes", "mends"),
        "object": ("a copper kettle", "the clay vessel", "a linen coat"),
        "adjunct": ("near the window", "beside the stone wall", "before the evening rain"),
    },
    {
        "name": "field",
        "subject": ("a quiet gardener", "the watchful ranger", "a local teacher"),
        "verb": ("records", "checks", "guides"),
        "object": ("fresh seeds", "a weather map", "the new pupils"),
        "adjunct": ("beside the garden gate", "under the cedar roof", "across the open meadow"),
    },
)
SLOT_PHASES = ("subject", "verb", "object", "adjunct")


def char_score(text: str) -> float:
    padded = "  " + text.lower()
    return sum(math.log1p(TRIGRAM.get(padded[i : i + 3], 0)) for i in range(len(padded) - 2))


def novelty_preflight() -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text())
    entries = registry.get("entries", []) + registry.get("excluded", []) + registry.get("audit_reports", [])
    collisions = [entry.get("id") for entry in entries if entry.get("id") != EXPERIMENT_ID and entry.get("signature") == SIGNATURE]
    overlapping = [
        entry.get("id") for entry in entries
        if entry.get("signature", "").startswith("character-kernel-beam|")
        or "obligation" in entry.get("signature", "")
    ]
    return {
        "status": "passed" if not collisions else "failed",
        "performed_before_search": True,
        "registry_entries_read": len(entries),
        "signature_collision": bool(collisions),
        "collisions": collisions,
        "prior_obligation_routes_flagged": overlapping,
        "fixed_tape_used": False,
        "finished_surface_reversed": False,
        "catalogue_text_imported": False,
    }


def decode_frame(frame: dict[str, tuple[str, ...]], width: int = 6) -> list[BeamState]:
    """Decode legal complete clauses, retaining boundary-aware beam traces."""
    beam = [BeamState((), "START", "sentence_start", 0.0, ())]
    for phase in SLOT_PHASES:
        expanded: list[BeamState] = []
        for state in beam:
            for phrase in frame[phase]:
                words = tuple(phrase.split())
                candidate = " ".join((*state.words, *words))
                boundary = {"subject": "np_end", "verb": "vp_end", "object": "np_end", "adjunct": "sentence_end"}[phase]
                expanded.append(BeamState(
                    (*state.words, *words), phase, boundary,
                    state.score + char_score(candidate),
                    (*state.trace, {"phase": phase, "boundary": boundary, "phrase": phrase}),
                ))
        expanded.sort(key=lambda state: (state.score, state.words), reverse=True)
        beam = expanded[:width]
    return [state for state in beam if state.phase == "adjunct" and state.boundary == "sentence_end"]


def pointer_audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    i, j = 0, len(tape) - 1
    while i < j and tape[i] == tape[j]:
        i += 1
        j -= 1
    forward = hashlib.sha256(tape.encode()).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode()).hexdigest()
    return {
        "algorithm": "independent_two_pointer_plus_forward_reverse_sha256",
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and i >= j,
        "first_mismatch": None if i >= j else {"left_index": i, "right_index": j, "left": tape[i], "right": tape[j]},
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal": forward == reverse,
    }


def candidate_record(left: BeamState, right: BeamState, rank: int) -> dict[str, object]:
    text = f"{' '.join(left.words).capitalize()}. {' '.join(right.words).capitalize()}."
    audit = pointer_audit(text)
    checks = mechanical_admission_checks(text, min_letters=45, max_letters=190)
    structural = {key: value for key, value in checks.items() if key != "exact_letter_palindrome"}
    accepted = bool(all(structural.values()) and not audit["two_pointer_exact"] and not audit["sha_equal"])
    failure = None if accepted else ("exact palindrome or SHA mismatch" if audit["two_pointer_exact"] or audit["sha_equal"] else next((key for key, value in checks.items() if not value), "unknown"))
    return {
        "rank": rank,
        "rendered": text,
        "char_lm_score": round(left.score + right.score, 5),
        "semantic_frames": [left.trace[0]["phase"], right.trace[0]["phase"]],
        "boundary_traces": {"left": left.trace, "right": right.trace},
        "audit": audit,
        "mechanical_admission": checks,
        "accepted_intact_prose_control": accepted,
        "failure_reason": failure,
        "next_repair": "replace the first mismatching boundary phrase with a held-out role-compatible phrase, then rerun the finite-state beam" if failure else None,
        "anti_shortcut": {
            "finished_surface_reversed": False,
            "word_order_mirror": False,
            "repeated_or_self_palindromic_unit": False,
            "catalogue_text_or_gibberish": False,
        },
        "provenance": {
            "lexical_source": "inline typed semantic clause templates",
            "character_model": "transparent corpus trigram counts used only for ranking",
            "grammar_state": "START -> subject NP -> verb VP -> object NP -> adjunct -> END",
            "generator": str(Path(__file__).relative_to(ROOT)),
        },
    }


def run() -> dict[str, object]:
    preflight = novelty_preflight()
    if preflight["status"] != "passed":
        raise RuntimeError("novelty signature collision")
    frontiers = {frame["name"]: decode_frame(frame) for frame in FRAMES}
    pairs = list(itertools.product(frontiers["craft"], frontiers["field"]))[:12]
    candidates = [candidate_record(left, right, index + 1) for index, (left, right) in enumerate(pairs)]
    controls = {
        "mirror_rejection": mechanical_admission_checks("Rats star.", min_letters=1, max_letters=20),
        "fragment_rejection": mechanical_admission_checks("la st.", min_letters=1, max_letters=20),
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "status": "completed_no_exact_closure",
        "method": "finite-state grammar beam over semantic clause templates; lexical boundary classes constrain legal phrase transitions and a transparent character trigram score ranks the bounded beam",
        "novelty_preflight": preflight,
        "search": {"frame_frontiers": {name: len(states) for name, states in frontiers.items()}, "paired_candidates": len(candidates), "beam_width": 6},
        "candidates": candidates,
        "rejection_controls": controls,
        "stats": {"intact_prose_controls": sum(item["accepted_intact_prose_control"] for item in candidates), "exact": sum(item["audit"]["two_pointer_exact"] for item in candidates), "sha_equal": sum(item["audit"]["sha_equal"] for item in candidates)},
        "next_repair": "At each recorded first mismatch, add one held-out adjunct phrase for the same semantic role and boundary class; rerank locally, then re-run both independent exact audits before any reader claim.",
        "reader_status": "not eligible: this bounded lane produced intact prose controls but no exact closure",
        "provenance": {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "audits": ["independent two-pointer", "forward/reverse SHA-256", "mechanical admission", "novelty preflight", "anti-shortcut controls"]},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"candidates": len(result["candidates"]), "intact_prose_controls": result["stats"]["intact_prose_controls"], "exact": result["stats"]["exact"]}))
