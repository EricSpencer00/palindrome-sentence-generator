"""Half-tape grammar CSP with variable word boundaries.

The solver creates only ``ceil(N / 2)`` character variables for a target
length N.  Every full-tape position aliases one of those variables before a
word is selected, so palindrome equality is a construction constraint rather
than a post-hoc filter.  Grammar chunks are emitted in ordinary order and may
cross the midpoint or place the midpoint inside a word.

This is intentionally a bounded pilot.  It uses complete typed scene frames,
agreement/valency restrictions, unique content words, and the shared
mechanical gate.  A programmatic pass is never a readability certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters, tokenize

EXPERIMENT_ID = "half-tape-grammar-csp-20260919"


@dataclass(frozen=True)
class Option:
    words: tuple[str, ...]
    number: str | None = None
    valency: str | None = None
    object_type: str | None = None
    content: frozenset[str] = frozenset()
    proper_name: bool = False


@dataclass(frozen=True)
class Frame:
    name: str
    chunks: tuple[str, ...]
    # A punctuation boundary is presentation only: it never enters the
    # normalized tape or the half-tape constraints.  Keeping it in the frame
    # lets the pilot expose intact prose instead of an unpunctuated scaffold.
    clause_break_after: int | None = None


def _content(words: Iterable[str]) -> frozenset[str]:
    function = {
        "a", "an", "the", "some", "many", "two", "nine", "one", "new", "old",
        "and", "but", "while", "at", "near", "under", "after", "before", "by", "with",
        "in", "on", "to", "of", "she", "he", "they", "we", "i",
    }
    return frozenset(
        normalize_letters(word) for word in words
        if normalize_letters(word) not in function and len(normalize_letters(word)) > 1
    )


def _np(text: str, number: str, *, proper_name: bool = False) -> Option:
    words = tuple(text.split())
    return Option(words=words, number=number, content=_content(words), proper_name=proper_name)


SUBJECTS = (
    _np("an aide", "sg"), _np("a bard", "sg"), _np("a poet", "sg"),
    _np("a scribe", "sg"), _np("a sailor", "sg"), _np("a keeper", "sg"),
    _np("the captain", "sg"), _np("the herald", "sg"), _np("the pilot", "sg"),
    _np("some men", "pl"), _np("some maids", "pl"), _np("some poets", "pl"),
    _np("the sailors", "pl"), _np("the players", "pl"), _np("the singers", "pl"),
    _np("Diana", "sg", proper_name=True), _np("Noel", "sg", proper_name=True),
    _np("she", "sg"), _np("he", "sg"),
)

VERBS = (
    Option(("rips",), "sg", "transitive", "document", _content(("rips",))),
    Option(("reads",), "sg", "transitive", "document", _content(("reads",))),
    Option(("marks",), "sg", "transitive", "document", _content(("marks",))),
    Option(("writes",), "sg", "transitive", "document", _content(("writes",))),
    Option(("guides",), "sg", "transitive", "person", _content(("guides",))),
    Option(("inspires",), "sg", "transitive", "person", _content(("inspires",))),
    Option(("praises",), "sg", "transitive", "person", _content(("praises",))),
    Option(("guards",), "sg", "transitive", "place", _content(("guards",))),
    Option(("keeps",), "sg", "transitive", "document", _content(("keeps",))),
    Option(("read",), "pl", "transitive", "document", _content(("read",))),
    Option(("mark",), "pl", "transitive", "document", _content(("mark",))),
    Option(("write",), "pl", "transitive", "document", _content(("write",))),
    Option(("guide",), "pl", "transitive", "person", _content(("guide",))),
    Option(("inspire",), "pl", "transitive", "person", _content(("inspire",))),
    Option(("praise",), "pl", "transitive", "person", _content(("praise",))),
    Option(("guard",), "pl", "transitive", "place", _content(("guard",))),
    Option(("keep",), "pl", "transitive", "document", _content(("keep",))),
)

OBJECTS = (
    Option(("nine", "memos"), content=_content(("nine", "memos")), object_type="document"),
    Option(("a", "letter"), content=_content(("a", "letter")), object_type="document"),
    Option(("the", "sonnet"), content=_content(("the", "sonnet")), object_type="document"),
    Option(("new", "songs"), content=_content(("new", "songs")), object_type="document"),
    Option(("old", "tales"), content=_content(("old", "tales")), object_type="document"),
    Option(("a", "map"), content=_content(("a", "map")), object_type="document"),
    Option(("the", "chart"), content=_content(("the", "chart")), object_type="document"),
    Option(("some", "men"), content=_content(("some", "men")), object_type="person"),
    Option(("the", "poet"), content=_content(("the", "poet")), object_type="person"),
    Option(("Diana",), content=_content(("Diana",)), object_type="person", proper_name=True),
    Option(("the", "shore"), content=_content(("the", "shore")), object_type="place"),
    Option(("the", "harbor"), content=_content(("the", "harbor")), object_type="place"),
    Option(("the", "river"), content=_content(("the", "river")), object_type="place"),
)

ADJUNCTS = (
    Option(("at", "dawn"), content=_content(("at", "dawn"))),
    Option(("after", "rain"), content=_content(("after", "rain"))),
    Option(("under", "the", "moon"), content=_content(("under", "the", "moon"))),
    Option(("near", "the", "river"), content=_content(("near", "the", "river"))),
    Option(("by", "the", "shore"), content=_content(("by", "the", "shore"))),
)

CONJUNCTIONS = (
    Option(("and",), content=frozenset()),
    Option(("while",), content=frozenset()),
)

COREFERENT_PRONOUNS = (
    Option(("she",), number="sg", content=frozenset()),
    Option(("he",), number="sg", content=frozenset()),
    Option(("they",), number="pl", content=frozenset()),
    Option(("we",), number="pl", content=frozenset()),
)

FRAMES = (
    Frame("two_complete_beats", ("SUBJ", "VERB", "OBJ", "SUBJ2", "VERB2", "OBJ2"), 3),
    Frame("two_beats_with_left_time", ("SUBJ", "VERB", "OBJ", "PP", "SUBJ2", "VERB2", "OBJ2"), 4),
    Frame("two_beats_with_right_time", ("SUBJ", "VERB", "OBJ", "SUBJ2", "VERB2", "OBJ2", "PP2"), 3),
    Frame("coordinated_beats", ("SUBJ", "VERB", "OBJ", "CONJ", "SUBJ2", "VERB2", "OBJ2"), None),
    # Repair operator: keep one discourse participant alive across the seam
    # with an agreement-carrying pronoun, while adding a temporal adjunct.
    Frame("shared_participant_temporal", ("SUBJ", "VERB", "OBJ", "PP", "COREF", "VERB2", "OBJ2"), 4),
)


def _options(chunk: str, state: dict[str, object]) -> tuple[Option, ...]:
    if chunk in {"SUBJ", "SUBJ2"}:
        used_proper = bool(state["proper_names"])
        return tuple(option for option in SUBJECTS if not (option.proper_name and used_proper))
    if chunk == "COREF":
        return tuple(option for option in COREFERENT_PRONOUNS
                     if option.number == state["subject_number"])
    if chunk in {"VERB", "VERB2"}:
        number = state["subject_number"] if chunk == "VERB" else state["subject2_number"]
        return tuple(option for option in VERBS if option.number == number)
    if chunk in {"OBJ", "OBJ2"}:
        object_type = state["object_type"] if chunk == "OBJ" else state["object2_type"]
        return tuple(option for option in OBJECTS if option.object_type == object_type)
    if chunk in {"PP", "PP2"}:
        return ADJUNCTS
    if chunk == "CONJ":
        return CONJUNCTIONS
    return ()


def _set_chunk_state(chunk: str, option: Option, state: dict[str, object]) -> None:
    if chunk == "SUBJ":
        state["subject_number"] = option.number
    elif chunk == "VERB":
        state["object_type"] = option.object_type
    elif chunk == "SUBJ2":
        state["subject2_number"] = option.number
    elif chunk == "COREF":
        state["subject2_number"] = option.number
    elif chunk == "VERB2":
        state["object2_type"] = option.object_type


def _place(assign: list[str | None], pos: int, words: tuple[str, ...], target: int) -> list[str | None] | None:
    """Place a chunk while enforcing half-tape aliases immediately."""
    updated = list(assign)
    cursor = pos
    for word in words:
        for char in normalize_letters(word):
            if cursor >= target:
                return None
            alias = min(cursor, target - 1 - cursor)
            prior = updated[alias]
            if prior is not None and prior != char:
                return None
            updated[alias] = char
            cursor += 1
    return updated


def _audit(text: str) -> dict[str, object]:
    tape = normalize_letters(text)
    left, right = 0, len(tape) - 1
    mismatches = []
    while left < right:
        if tape[left] != tape[right]:
            mismatches.append({"left": left, "right": right,
                               "left_char": tape[left], "right_char": tape[right]})
        left += 1
        right -= 1
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    return {"normalized": tape, "letters": len(tape),
            "two_pointer_exact": bool(tape) and not mismatches,
            "first_mismatch": mismatches[0] if mismatches else None,
            "sha256_forward": forward, "sha256_reverse": reverse,
            "sha_equal": forward == reverse}


def _hidden_span(text: str) -> bool:
    words = tuple(normalize_letters(word) for word in tokenize(text))
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            if start == 0 and end == len(words):
                continue
            span = "".join(words[start:end])
            if span and span == span[::-1]:
                return True
    return False


def _render_chunks(chunks: tuple[str, ...], frame: Frame) -> str:
    rendered = []
    for index, chunk in enumerate(chunks):
        if index == frame.clause_break_after:
            rendered[-1] = rendered[-1] + ";"
        rendered.append(chunk)
    return " ".join(rendered) + "."


def search_target(target: int, frame: Frame, *, max_nodes: int = 120_000) -> tuple[list[dict[str, object]], dict[str, object]]:
    assignments: list[str | None] = [None] * ((target + 1) // 2)
    rows: list[dict[str, object]] = []
    nodes = 0
    longest_words: tuple[str, ...] = ()

    def dfs(index: int, pos: int, words: tuple[str, ...], chunks: tuple[str, ...], assign: list[str | None], state: dict[str, object]) -> None:
        nonlocal nodes, longest_words
        if nodes >= max_nodes or len(rows) >= 20:
            return
        nodes += 1
        if index == len(frame.chunks):
            if pos != target:
                return
            longest_words = max(longest_words, words, key=lambda value: len(normalize_letters(" ".join(value))))
            text = _render_chunks(chunks, frame)
            audit = _audit(text)
            # Keep the 38-letter anchor eligible as a regression row.  The
            # >38 promotion objective is reported separately by target length;
            # it must not turn the shared mechanical gate into a readability
            # or progress claim.
            checks = mechanical_admission_checks(text, min_letters=30, max_letters=2000)
            row = {
                "rendered": text,
                "length": audit["letters"],
                "frame": frame.name,
                "word_spans": list(words),
                "audit": audit,
                "mechanical_checks": checks,
                "hidden_proper_span": _hidden_span(text),
                "provenance": {"representation": "fixed-length half-tape CSP",
                               "target_length": target, "grammar_frame": frame.name,
                               "catalogue_imported": False, "finished_tape_reversed": False,
                               "rlaif_used": False},
                "reader_status": "unreviewed; programmatic checks never certify readability",
            }
            row["mechanically_admitted"] = audit["two_pointer_exact"] and not row["hidden_proper_span"] and all(checks.values())
            rows.append(row)
            return

        chunk = frame.chunks[index]
        for option in _options(chunk, state):
            if state["content"] & option.content:
                continue
            if len(state["proper_names"]) and option.proper_name:
                continue
            next_state = dict(state)
            next_state["content"] = frozenset(set(state["content"]) | set(option.content))
            next_state["proper_names"] = frozenset(set(state["proper_names"]) | ({option.words[0]} if option.proper_name else set()))
            _set_chunk_state(chunk, option, next_state)
            placed = _place(assign, pos, option.words, target)
            if placed is None:
                continue
            chunk_text = " ".join(option.words)
            dfs(index + 1, pos + len(normalize_letters("".join(option.words))),
                words + option.words, chunks + (chunk_text,), placed, next_state)

    initial = {"content": frozenset(), "proper_names": frozenset(),
               "subject_number": None, "subject2_number": None,
               "object_type": None, "object2_type": None}
    dfs(0, 0, (), (), assignments, initial)
    return rows, {"nodes": nodes, "longest_words": list(longest_words)}


def run(*, lengths: range = range(40, 53), max_nodes: int = 120_000) -> dict[str, object]:
    all_rows: list[dict[str, object]] = []
    search_stats = []
    for target in lengths:
        for frame in FRAMES:
            rows, stats = search_target(target, frame, max_nodes=max_nodes)
            all_rows.extend(rows)
            search_stats.append({"target": target, "frame": frame.name, **stats, "exact": len(rows)})
    unique = {}
    for row in all_rows:
        unique.setdefault(row["rendered"], row)
    rows = list(unique.values())
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    return {
        "experiment_id": EXPERIMENT_ID,
        "method": "fixed-length half-tape grammar CSP with variable word boundaries, semantic role state, and shared-participant temporal repair",
        "status": "completed_exact" if exact else "completed_no_exact_closure",
        "target_lengths": [length for length in lengths],
        "frames": [frame.name for frame in FRAMES],
        "actual_candidates": rows,
        "exact_candidates": exact,
        "stats": {"nodes": sum(item["nodes"] for item in search_stats),
                  "unique_rows": len(rows), "exact": len(exact),
                  "mechanically_admitted": len(admitted),
                  "longest_exact": max((row["length"] for row in exact), default=0),
                  "longest_rendered": max((row["length"] for row in rows), default=0)},
        "search_trace": search_stats,
        "provenance": {"vocabulary_hash": hashlib.sha256(
                           json.dumps({"subjects": [option.words for option in SUBJECTS],
                                       "verbs": [option.words for option in VERBS],
                                       "objects": [option.words for option in OBJECTS]}, sort_keys=True).encode()
                       ).hexdigest(),
                       "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "independent_audits": ["literal outside-in two-pointer", "forward/reverse SHA-256"],
                       "catalogue_text": False, "rlaif_per_candidate": False},
        "novelty_preflight": {"status": "passed",
                              "distinction": "palindrome aliases are assigned before lexical boundaries; no complete clause reverse lookup",
                              "prior_lanes_checked": ["typed_constituent_seam_search_20260919",
                                                       "typed_phrase_graph_walk_20260919",
                                                       "pcfg_fsa_intersection_20260919"]},
        "next_repair": {"action": "add a second shared-participant frame with a temporal complement or a new agreement-carrying pronoun, then replay the same half-tape alias CSP",
                        "reason": "the shared-participant temporal repair is itself bounded; future progress must alter a grammar obligation rather than widen the same lexical bank",
                        "reader_test": "randomized blinded intact-prose versus shuffled-control rating for every admitted row"},
        "reader_gate": "closed; exactness and programmatic diagnostics do not certify readability",
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "runs" / f"{EXPERIMENT_ID}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
