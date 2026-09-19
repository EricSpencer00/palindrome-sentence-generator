"""Two-sided exact construction from independent finite semantic paths.

This experiment is deliberately a constructor rather than a repair pass.  A
complete finite scene path is chosen from each side's ordinary grammar, both
paths are compiled into character tries, and a product transition assigns one
mirrored character orbit before either path can reach its next character.  The
right path is traversed backwards only as an index; it is always rendered in
its original grammatical order.

No completed palindrome, reversible lexical-pair table, word-order mirror,
catalogue sentence, language-model reward, or post-hoc edit is an input.  A
product terminal is a candidate only when both sides are complete semantic
SVO paths.  Intact complete-scene controls are retained when the bounded
product exhausts without a closure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
from collections import Counter
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_palindrome.admission import mechanical_admission_checks, normalize_letters


EXPERIMENT_ID = "two-sided-semantic-orbit-product-20260920"
SIGNATURE = (
    "independent-semantic-scene-paths|two-sided-character-orbit-product|"
    "complete-svo-terminal-gate|ordinary-right-rendering"
)
# The bounded frontier is deliberately the same 39--60-letter band used by
# the current exact-construction acceptance work.  Keeping it here (rather
# than passing an arbitrary range to the constructor) makes the run and its
# evidence artifact reproducible.
MIN_TOTAL_LETTERS = 39
MAX_TOTAL_LETTERS = 60
DEFAULT_MAX_PATHS = 12_000
DEFAULT_MAX_STATES = 100_000


@dataclass(frozen=True)
class Segment:
    """One grammar terminal with semantic features kept beside its text."""

    role: str
    text: str
    number: str = ""
    kind: str = ""
    semantic_id: str = ""

    @property
    def tape(self) -> str:
        return normalize_letters(self.text)


@dataclass(frozen=True)
class ScenePath:
    """A complete ordinary-order finite semantic path."""

    path_id: str
    side: str
    frame: str
    segments: tuple[Segment, ...]

    @property
    def text(self) -> str:
        return " ".join(segment.text for segment in self.segments)

    @property
    def tape(self) -> str:
        return "".join(segment.tape for segment in self.segments)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(segment.role for segment in self.segments)

    @property
    def char_roles(self) -> tuple[str, ...]:
        """Grammar role carried for every character in the path tape."""
        return tuple(
            role
            for segment in self.segments
            for role in (segment.role,) * len(segment.tape)
        )

    @property
    def char_boundaries(self) -> tuple[str, ...]:
        """Boundary-aware grammar state carried for every character.

        Spaces are not part of the palindrome tape, so a plain character trie
        would otherwise lose the fact that a transition crossed from, say,
        ``SUBJECT`` into ``FINITE_VERB``.  Segment roles are fixed when a path
        is selected and travel with every character into the product state.
        """
        return tuple(
            f"{index}:{segment.role}"
            for index, segment in enumerate(self.segments)
            for _ in segment.tape
        )

    @property
    def word_boundaries(self) -> tuple[dict[str, object], ...]:
        """Return the ordinary rendering boundaries selected before search."""
        offset = 0
        boundaries: list[dict[str, object]] = []
        for index, segment in enumerate(self.segments):
            end = offset + len(segment.tape)
            boundaries.append({
                "segment_index": index,
                "role": segment.role,
                "text": segment.text,
                "start": offset,
                "end": end,
            })
            offset = end
        return tuple(boundaries)

    @property
    def content_words(self) -> frozenset[str]:
        repeatable = {
            "a", "an", "the", "some", "and", "while", "as", "at", "by",
            "in", "near", "after", "before", "under", "through", "of", "to",
        }
        return frozenset(
            word for word in self.tape_words
            if word not in repeatable
        )

    @property
    def tape_words(self) -> tuple[str, ...]:
        return tuple(normalize_letters(word) for word in self.text.split())

    @property
    def complete_finite_semantics(self) -> bool:
        roles = set(self.roles)
        return {"SUBJECT", "FINITE_VERB", "OBJECT"}.issubset(roles)

    def provenance(self) -> dict[str, object]:
        return {
            "path_id": self.path_id,
            "side": self.side,
            "frame": self.frame,
            "complete_finite_semantic_path": self.complete_finite_semantics,
            "word_boundaries": list(self.word_boundaries),
            "segments": [
                {
                    "role": segment.role,
                    "text": segment.text,
                    "number": segment.number,
                    "kind": segment.kind,
                    "semantic_id": segment.semantic_id,
                }
                for segment in self.segments
            ],
        }


# The two banks are authored independently.  They intentionally have no
# positional pairing: the product sees grammar states and characters only.
# All entries are ordinary lower-case lexical forms; no catalogue text is
# loaded as a generation source.
LEFT_BANK = {
    "subjects": (
        ("baker", "sg", "person"), ("captain", "sg", "person"),
        ("carer", "sg", "person"), ("clerk", "sg", "person"),
        ("farmer", "sg", "person"), ("guard", "sg", "person"),
        ("poet", "sg", "person"), ("reader", "sg", "person"),
        ("sailor", "sg", "person"), ("singer", "sg", "person"),
        ("scribe", "sg", "person"), ("teacher", "sg", "person"),
        ("usher", "sg", "person"), ("writer", "sg", "person"),
        ("bakers", "pl", "person"), ("clerks", "pl", "person"),
        ("farmers", "pl", "person"), ("poets", "pl", "person"),
        ("sailors", "pl", "person"), ("writers", "pl", "person"),
    ),
    "objects": (
        ("book", "sg", "thing"), ("chart", "sg", "thing"),
        ("letter", "sg", "thing"), ("map", "sg", "thing"),
        ("memo", "sg", "thing"), ("note", "sg", "thing"),
        ("page", "sg", "thing"), ("poem", "sg", "thing"),
        ("plan", "sg", "thing"), ("parcel", "sg", "thing"),
        ("song", "sg", "thing"), ("tale", "sg", "thing"),
        ("books", "pl", "thing"), ("charts", "pl", "thing"),
        ("letters", "pl", "thing"), ("maps", "pl", "thing"),
        ("notes", "pl", "thing"), ("pages", "pl", "thing"),
        ("poems", "pl", "thing"), ("plans", "pl", "thing"),
    ),
    "verbs": (
        ("carries", "carry", "thing"), ("draws", "draw", "thing"),
        ("finds", "find", "thing"), ("guards", "guard", "thing"),
        ("keeps", "keep", "thing"), ("marks", "mark", "thing"),
        ("mends", "mend", "thing"), ("opens", "open", "thing"),
        ("reads", "read", "thing"), ("records", "record", "thing"),
        ("saves", "save", "thing"), ("sends", "send", "thing"),
        ("signs", "sign", "thing"), ("weaves", "weave", "thing"),
        ("writes", "write", "thing"),
    ),
    "places": (
        ("dawn", "time"), ("dusk", "time"), ("harbor", "place"),
        ("river", "place"), ("shore", "place"), ("gate", "place"),
        ("garden", "place"), ("moon", "place"),
    ),
}

RIGHT_BANK = {
    "subjects": (
        ("artist", "sg", "person"), ("author", "sg", "person"),
        ("carpenter", "sg", "person"), ("chorister", "sg", "person"),
        ("driver", "sg", "person"), ("editor", "sg", "person"),
        ("healer", "sg", "person"), ("hunter", "sg", "person"),
        ("keeper", "sg", "person"), ("librarian", "sg", "person"),
        ("porter", "sg", "person"), ("reader", "sg", "person"),
        ("sailor", "sg", "person"), ("student", "sg", "person"),
        ("artists", "pl", "person"), ("authors", "pl", "person"),
        ("drivers", "pl", "person"), ("editors", "pl", "person"),
        ("keepers", "pl", "person"), ("porters", "pl", "person"),
    ),
    "objects": (
        ("archive", "sg", "place"), ("bell", "sg", "thing"),
        ("bridge", "sg", "place"), ("garden", "sg", "place"),
        ("harbor", "sg", "place"), ("ledger", "sg", "thing"),
        ("message", "sg", "thing"), ("river", "sg", "place"),
        ("road", "sg", "place"), ("scroll", "sg", "thing"),
        ("shore", "sg", "place"), ("story", "sg", "thing"),
        ("tower", "sg", "place"), ("archives", "pl", "place"),
        ("bells", "pl", "thing"), ("bridges", "pl", "place"),
        ("gardens", "pl", "place"), ("ledgers", "pl", "thing"),
        ("messages", "pl", "thing"), ("roads", "pl", "place"),
    ),
    "verbs": (
        ("admires", "admire", "person"), ("answers", "answer", "thing"),
        ("follows", "follow", "place"), ("greets", "greet", "person"),
        ("hears", "hear", "thing"), ("helps", "help", "person"),
        ("joins", "join", "person"), ("loves", "love", "person"),
        ("meets", "meet", "person"), ("praises", "praise", "person"),
        ("sees", "see", "place"), ("teaches", "teach", "person"),
        ("visits", "visit", "place"), ("watches", "watch", "thing"),
        ("welcomes", "welcome", "person"),
    ),
    "places": (
        ("noon", "time"), ("winter", "time"), ("shore", "place"),
        ("tower", "place"), ("village", "place"), ("arch", "place"),
        ("river", "place"), ("stars", "place"),
    ),
}

SUBJECT_ADJECTIVES = ("patient", "careful", "quiet", "watchful")
OBJECT_ADJECTIVES = ("old", "bright", "sealed", "small")
DETERMINERS = {"sg": ("a", "the"), "pl": ("some", "the")}
PREPOSITIONS = ("at", "near", "by", "after", "before", "under", "through")


def _article(noun: str) -> str:
    return "an" if noun[0] in "aeiou" else "a"


def _subject_segments(noun: tuple[str, str, str], adjective: str | None = None) -> tuple[Segment, ...]:
    word, number, kind = noun
    determiner = _article(adjective or word) if number == "sg" else "some"
    segments = [Segment("SUBJECT_DET", determiner, number, kind, f"det:{determiner}")]
    if adjective:
        segments.append(Segment("SUBJECT_ADJ", adjective, number, kind, f"adj:{adjective}"))
    segments.append(Segment("SUBJECT", word, number, kind, f"subject:{word}"))
    return tuple(segments)


def _object_segments(noun: tuple[str, str, str], adjective: str | None = None) -> tuple[Segment, ...]:
    word, number, kind = noun
    determiner = _article(adjective or word) if number == "sg" else "some"
    segments = [Segment("OBJECT_DET", determiner, number, kind, f"det:{determiner}")]
    if adjective:
        segments.append(Segment("OBJECT_ADJ", adjective, number, kind, f"adj:{adjective}"))
    segments.append(Segment("OBJECT", word, number, kind, f"object:{word}"))
    return tuple(segments)


def _verb_text(singular: str, plural: str, number: str) -> str:
    return singular if number == "sg" else plural


def build_paths(
    side: str,
    *,
    max_paths: int = DEFAULT_MAX_PATHS,
    max_letters: int = 40,
) -> tuple[ScenePath, ...]:
    """Enumerate bounded complete story paths before any orbit is expanded.

    The path inventory is intentionally stratified across four ordinary
    clause frames.  A prior prototype filled its cap with one long frame,
    making the 39--60-letter product unnecessarily large while hiding the
    grammar boundary choices.  Each frame now gets its own deterministic
    quota, and overlong paths are rejected before they enter a trie.
    """
    bank = LEFT_BANK if side == "left" else RIGHT_BANK
    paths: list[ScenePath] = []
    path_number = 0
    # Separate quotas keep the story grammar present in the finite product.
    frames = ("svo", "svo_subject_adj", "svo_object_adj", "svo_adjunct")
    frame_limits = {frame: max(1, max_paths // len(frames)) for frame in frames}
    frame_counts = Counter()

    for frame in frames:
        for subject in bank["subjects"]:
            subject_adjectives = SUBJECT_ADJECTIVES if frame == "svo_subject_adj" else (None,)
            for subject_adj in subject_adjectives:
                subject_segments = _subject_segments(subject, subject_adj)
                for singular, plural, verb_kind in bank["verbs"]:
                    verb = _verb_text(singular, plural, subject[1])
                    verb_segment = Segment("FINITE_VERB", verb, subject[1], verb_kind, f"verb:{singular}")
                    for obj in bank["objects"]:
                        if obj[2] != verb_kind:
                            continue
                        object_adjectives = OBJECT_ADJECTIVES if frame == "svo_object_adj" else (None,)
                        for object_adj in object_adjectives:
                            object_segments = _object_segments(obj, object_adj)
                            base_segments = subject_segments + (verb_segment,) + object_segments
                            adjunct_options: Iterable[tuple[Segment, ...]] = ((),)
                            if frame == "svo_adjunct":
                                adjunct_options = (
                                    (
                                        Segment("ADJUNCT_PREP", prep, semantic_id=f"prep:{prep}"),
                                        Segment("ADJUNCT_OBJECT", place, kind=place_kind, semantic_id=f"place:{place}"),
                                    )
                                    for prep in PREPOSITIONS
                                    for place, place_kind in bank["places"]
                                )
                            for adjunct in adjunct_options:
                                segments = base_segments + tuple(adjunct)
                                content = [
                                    normalize_letters(segment.text)
                                    for segment in segments
                                    if segment.role not in {"SUBJECT_DET", "OBJECT_DET", "ADJUNCT_PREP"}
                                ]
                                if len(content) != len(set(content)):
                                    continue
                                path = ScenePath(
                                    path_id=f"{side}-{frame}-{path_number:06d}",
                                    side=side,
                                    frame=frame,
                                    segments=segments,
                                )
                                if len(path.tape) > max_letters:
                                    continue
                                paths.append(path)
                                path_number += 1
                                frame_counts[frame] += 1
                                if frame_counts[frame] >= frame_limits[frame]:
                                    break
                            if frame_counts[frame] >= frame_limits[frame]:
                                break
                        if frame_counts[frame] >= frame_limits[frame]:
                            break
                    if frame_counts[frame] >= frame_limits[frame]:
                        break
                if frame_counts[frame] >= frame_limits[frame]:
                    break
            if frame_counts[frame] >= frame_limits[frame]:
                break
    # Stable ordering makes hashes and frontier samples reproducible across
    # hosts while retaining the grammar-frame stratification.
    paths.sort(key=lambda path: (path.frame, len(path.tape), path.path_id))
    return tuple(paths)


@dataclass
class TrieNode:
    children: dict[str, int] = field(default_factory=dict)
    terminals: list[int] = field(default_factory=list)
    role_states: set[str] = field(default_factory=set)
    boundary_states: set[str] = field(default_factory=set)


class PathTrie:
    """Character trie preserving role and word-boundary state at every edge.

    ``reverse_cursor`` is a grammar cursor orientation, not a rendered-tape
    transform.  The left trie is walked from the final character toward the
    clause centre and the right trie from its first character outward; both
    paths are rendered in their original ordinary order.
    """

    def __init__(self, paths: Iterable[ScenePath], *, reverse_cursor: bool):
        self.paths = tuple(paths)
        self.reverse_cursor = reverse_cursor
        self.nodes = [TrieNode()]
        for path_index, path in enumerate(self.paths):
            tape = path.tape
            roles = path.char_roles
            boundaries = path.char_boundaries
            stream = tape[::-1] if reverse_cursor else tape
            stream_roles = roles[::-1] if reverse_cursor else roles
            stream_boundaries = boundaries[::-1] if reverse_cursor else boundaries
            node_index = 0
            for character, role, boundary in zip(stream, stream_roles, stream_boundaries):
                node = self.nodes[node_index]
                node.role_states.add(role)
                node.boundary_states.add(boundary)
                child_index = node.children.get(character)
                if child_index is None:
                    child_index = len(self.nodes)
                    self.nodes.append(TrieNode())
                    node.children[character] = child_index
                node_index = child_index
            self.nodes[node_index].role_states.update(roles[:1] if reverse_cursor else roles[-1:])
            self.nodes[node_index].boundary_states.update(boundaries[:1] if reverse_cursor else boundaries[-1:])
            self.nodes[node_index].terminals.append(path_index)


def independent_audit(text: str) -> dict[str, object]:
    """Independent two-pointer and forward/reverse SHA-256 checks."""
    tape = normalize_letters(text)
    left, right = 0, len(tape) - 1
    mismatch: dict[str, object] | None = None
    while left < right:
        if tape[left] != tape[right]:
            mismatch = {
                "orbit": left,
                "left_index": left,
                "right_index": right,
                "left_char": tape[left],
                "right_char": tape[right],
            }
            break
        left += 1
        right -= 1
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    backward = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    return {
        "normalized": tape,
        "letters": len(tape),
        "two_pointer_exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
        "sha256_forward": forward,
        "sha256_reverse": backward,
        "sha256_equal": forward == backward,
    }


def render_pair(left: ScenePath, right: ScenePath) -> str:
    """Render both ordinary paths; no reverse or word-order transform occurs."""
    left_text = left.text[:1].upper() + left.text[1:]
    return f"{left_text}; {right.text}."


def _candidate_row(
    left: ScenePath,
    right: ScenePath,
    *,
    source: str,
    orbit: dict[str, object] | None = None,
) -> dict[str, object]:
    rendered = render_pair(left, right)
    audit = independent_audit(rendered)
    checks = mechanical_admission_checks(
        rendered,
        min_letters=MIN_TOTAL_LETTERS,
        max_letters=MAX_TOTAL_LETTERS,
    )
    return {
        "rendered": rendered,
        "source": source,
        "orbit_assignment": orbit or {
            "constructed_before_render": False,
            "first_failure": None,
        },
        "complete_paths": {
            "left": left.provenance(),
            "right": right.provenance(),
        },
        "audit": audit,
        "mechanical_checks": checks,
        "mechanically_admitted": bool(audit["two_pointer_exact"]) and all(checks.values()),
        "provenance": {
            "construction": "independent finite semantic story paths joined by a live center-out character-orbit product",
            "lexical_banks_independent": True,
            "grammar_boundaries_selected_before_render": True,
            "semantic_roles_selected_before_render": True,
            "center_out_cursor": "left clause end and right clause start",
            "right_path_rendered_in_ordinary_order": True,
            "completed_palindrome_seed": False,
            "reversible_lexical_pairs_used": False,
            "word_order_mirror": False,
            "finished_tape_reversal": False,
            "repair_mismatch_after_render": False,
            "catalogue_text_imported": False,
            "rlaif_used": False,
            "reader_status": "unreviewed; exactness and mechanical checks are not readability evidence",
        },
    }


def _orbit_product(
    left_trie: PathTrie,
    right_trie: PathTrie,
    *,
    min_letters: int = MIN_TOTAL_LETTERS,
    max_letters: int = MAX_TOTAL_LETTERS,
    max_states: int = DEFAULT_MAX_STATES,
    max_pair_rows: int = 1_000,
) -> dict[str, object]:
    """Expand complete grammar states from the story centre outward.

    The ordinary left path is indexed backwards from its final character and
    the ordinary right path forwards from its first character.  A common edge
    assigns exactly one mirrored orbit before either side can advance.  The
    optional ``left_center``/``right_center`` starts consume one unpaired
    centre character, which allows odd-length palindromes without padding or
    repairing a finished tape.
    """
    # ``mode`` is even, left_center, or right_center.  ``depth`` counts paired
    # orbits only; the centre character, when present, is tracked by mode.
    stack: list[tuple[int, int, int, str]] = [(0, 0, 0, "even")]
    root_left = left_trie.nodes[0]
    root_right = right_trie.nodes[0]
    stack.extend((child, 0, 0, "left_center") for child in root_left.children.values())
    stack.extend((0, child, 0, "right_center") for child in root_right.children.values())
    visited: set[tuple[int, int, int, str]] = set()
    closure_pairs: list[dict[str, object]] = []
    closure_seen: set[tuple[int, int, str]] = set()
    dead_states: Counter[int] = Counter()
    dead_samples: list[dict[str, object]] = []
    expanded = 0
    matched = 0
    rejected = 0
    max_depth = 0
    budget_exhausted = False
    while stack:
        left_index, right_index, depth, mode = stack.pop()
        state_key = (left_index, right_index, depth, mode)
        if state_key in visited:
            continue
        visited.add(state_key)
        expanded += 1
        max_depth = max(max_depth, depth)
        if expanded > max_states:
            budget_exhausted = True
            break
        left_node = left_trie.nodes[left_index]
        right_node = right_trie.nodes[right_index]
        total = 2 * depth + (0 if mode == "even" else 1)
        if left_node.terminals and right_node.terminals and min_letters <= total <= max_letters:
            for left_path_index in left_node.terminals:
                for right_path_index in right_node.terminals:
                    pair = (left_path_index, right_path_index, mode)
                    if pair not in closure_seen:
                        closure_seen.add(pair)
                        closure_pairs.append({
                            "left": left_path_index,
                            "right": right_path_index,
                            "center_mode": mode,
                            "paired_orbits": depth,
                            "letters": total,
                        })
                        if len(closure_pairs) >= max_pair_rows:
                            break
                if len(closure_pairs) >= max_pair_rows:
                    break
        next_total = 2 * (depth + 1) + (0 if mode == "even" else 1)
        if next_total > max_letters:
            continue
        common = sorted(set(left_node.children).intersection(right_node.children))
        rejected += len(set(left_node.children).symmetric_difference(right_node.children))
        if not common:
            dead_states[depth] += 1
            if len(dead_samples) < 12:
                dead_samples.append({
                    "orbit_depth": depth,
                    "center_mode": mode,
                    "total_letters_so_far": total,
                    "left_roles": sorted(left_node.role_states),
                    "right_roles": sorted(right_node.role_states),
                    "left_boundaries": sorted(left_node.boundary_states),
                    "right_boundaries": sorted(right_node.boundary_states),
                    "left_next_characters": sorted(left_node.children),
                    "right_next_characters": sorted(right_node.children),
                    "left_terminal": bool(left_node.terminals),
                    "right_terminal": bool(right_node.terminals),
                })
            continue
        for character in common:
            matched += 1
            stack.append((left_node.children[character], right_node.children[character], depth + 1, mode))
    return {
        "expanded_states": expanded,
        "visited_states": len(visited),
        "matched_orbit_transitions": matched,
        "rejected_orbit_transitions": rejected,
        "max_orbit_depth": max_depth,
        "budget_exhausted": budget_exhausted,
        "closure_pairs": closure_pairs,
        "dead_states_by_orbit": {str(depth): count for depth, count in sorted(dead_states.items())},
        "dead_frontier_samples": dead_samples,
    }


def _shared_center_orbits(left: ScenePath, right: ScenePath) -> tuple[int, dict[str, object] | None]:
    """Compare centre-out cursors without constructing a reversed tape."""
    limit = min(len(left.tape), len(right.tape))
    for orbit in range(limit):
        left_index = len(left.tape) - 1 - orbit
        right_index = orbit
        left_char = left.tape[left_index]
        right_char = right.tape[right_index]
        if left_char != right_char:
            return orbit, {
                "orbit": orbit,
                "left_index": left_index,
                "right_index": right_index,
                "left_char": left_char,
                "right_char": right_char,
                "left_role": left.char_roles[left_index],
                "right_role": right.char_roles[right_index],
                "left_boundary": left.char_boundaries[left_index],
                "right_boundary": right.char_boundaries[right_index],
            }
    return limit, None


def _control_pairs(
    left_paths: tuple[ScenePath, ...],
    right_paths: tuple[ScenePath, ...],
    count: int = 4,
) -> list[tuple[ScenePath, ScenePath]]:
    """Retain intact complete paths at the strongest live centre frontier.

    Pairing is intentionally bounded; this is evidence preservation, not a
    second exhaustive search after the product.  The strongest rows are
    selected by the number of already assigned centre orbits, then by length.
    """
    controls: list[tuple[ScenePath, ScenePath]] = []
    scored: list[tuple[int, int, int, ScenePath, ScenePath]] = []
    for left in left_paths[:800]:
        for right in right_paths[:800]:
            total = len(left.tape) + len(right.tape)
            if not MIN_TOTAL_LETTERS <= total <= MAX_TOTAL_LETTERS:
                continue
            if left.content_words.intersection(right.content_words):
                continue
            rendered = render_pair(left, right)
            audit = independent_audit(rendered)
            if audit["two_pointer_exact"]:
                continue
            shared, _ = _shared_center_orbits(left, right)
            scored.append((shared, total, len(left.tape), left, right))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3].path_id, item[4].path_id))
    seen: set[str] = set()
    for _, _, _, left, right in scored:
        key = f"{left.path_id}|{right.path_id}"
        if key in seen:
            continue
        seen.add(key)
        controls.append((left, right))
        if len(controls) >= count:
            break
    return controls


def _registry_preflight() -> dict[str, object]:
    path = ROOT / "docs" / "experiment-novelty-registry.json"
    entries = 0
    signatures: list[str] = []
    try:
        payload = json.loads(path.read_text())
        rows = payload.get("entries", [])
        entries = len(rows)
        signatures = [str(row.get("signature", "")) for row in rows if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return {"status": "failed", "registry_entries_read": 0, "reason": "registry unavailable"}
    # Once this committed run is entered in the registry, replaying the same
    # deterministic artifact must remain reproducible.  A signature collision
    # from a different experiment is still a hard stop; this lane's own
    # registration is not a duplicate sweep.
    collision = any(
        signature == SIGNATURE
        and str(row.get("id", "")) != EXPERIMENT_ID
        for row in rows
        if isinstance(row, dict)
        for signature in [str(row.get("signature", ""))]
    )
    return {
        "status": "failed" if collision else "passed",
        "registry_entries_read": entries,
        "signature_collision": collision,
        "signature": SIGNATURE,
        "catalogue_text_used_as_input": False,
        "known_catalogue_used_only_by_shared_exclusion_gate": True,
        "reversible_lexical_pairs_used": False,
        "word_order_mirror_used": False,
        "finished_tape_reversal_used": False,
        "rlaif_used": False,
    }


def run(*, max_paths: int = DEFAULT_MAX_PATHS, max_states: int = DEFAULT_MAX_STATES) -> dict[str, object]:
    preflight = _registry_preflight()
    if preflight["status"] != "passed":
        raise RuntimeError("novelty preflight failed before search")
    # The target band bounds each side once the minimum path lengths are
    # known.  A generous 40-letter path cap keeps the finite story bank useful
    # for controls while preventing an accidental million-path construction.
    left_paths = build_paths("left", max_paths=max_paths, max_letters=40)
    right_paths = build_paths("right", max_paths=max_paths, max_letters=40)
    # True centre-out orientation: left moves from its ordinary end, right
    # from its ordinary beginning.  Rendering below never reverses either
    # completed path.
    left_trie = PathTrie(left_paths, reverse_cursor=True)
    right_trie = PathTrie(right_paths, reverse_cursor=False)
    product = _orbit_product(left_trie, right_trie, max_states=max_states)
    rows: list[dict[str, object]] = []
    duplicate_rejections = 0
    mechanical_rejections = 0
    seen_tapes: set[str] = set()
    for closure in product["closure_pairs"]:
        left = left_paths[int(closure["left"])]
        right = right_paths[int(closure["right"])]
        row = _candidate_row(
            left,
            right,
            source="live-terminal-semantic-orbit-closure",
            orbit={
                "constructed_before_render": True,
                "center_mode": closure["center_mode"],
                "paired_orbits": closure["paired_orbits"],
                "letters": closure["letters"],
                "first_failure": None,
            },
        )
        if not row["audit"]["two_pointer_exact"]:
            raise AssertionError("orbit product emitted a non-exact terminal")
        if not all(row["mechanical_checks"].values()):
            mechanical_rejections += 1
            # Keep the actual rendered exact row in the evidence artifact; the
            # mechanical gate must explain rejection rather than erase it.
        normalized = row["audit"]["normalized"]
        if normalized in seen_tapes:
            duplicate_rejections += 1
            continue
        seen_tapes.add(normalized)
        rows.append(row)
    controls = []
    for left, right in _control_pairs(left_paths, right_paths):
        shared, failure = _shared_center_orbits(left, right)
        controls.append(_candidate_row(
            left,
            right,
            source="intact-complete-semantic-story-control",
            orbit={
                "constructed_before_render": False,
                "center_mode": "control",
                "assigned_orbits": shared,
                "first_failure": failure,
            },
        ))
    exact = [row for row in rows if row["audit"]["two_pointer_exact"]]
    admitted = [row for row in exact if row["mechanically_admitted"]]
    all_rendered = rows + controls
    return {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": "fresh center-out story grammar; each accepted edge assigns one mirrored character orbit",
        "target_range": [MIN_TOTAL_LETTERS, MAX_TOTAL_LETTERS],
        "run_config": {
            "max_paths_per_side": max_paths,
            "max_product_states": max_states,
            "path_length_cap": 40,
            "remote_bounded_run_required": True,
        },
        "execution": {
            "host": socket.gethostname(),
            "runtime": "deterministic Python; no model or remote API",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lexical_bank_sha256": hashlib.sha256(
                json.dumps({"left": LEFT_BANK, "right": RIGHT_BANK}, sort_keys=True).encode()
            ).hexdigest(),
        },
        "grammar": {
            "left_complete_path_frames": ["SVO", "SVO+subject-adjective", "SVO+object-adjective", "SVO+adjunct"],
            "right_complete_path_frames": ["SVO", "SVO+subject-adjective", "SVO+object-adjective", "SVO+adjunct"],
            "required_roles": ["SUBJECT", "FINITE_VERB", "OBJECT"],
            "semantic_roles": [
                "SUBJECT_DET", "SUBJECT_ADJ", "SUBJECT", "FINITE_VERB",
                "OBJECT_DET", "OBJECT_ADJ", "OBJECT", "ADJUNCT_PREP", "ADJUNCT_OBJECT",
            ],
            "boundary_policy": "complete segment boundaries are selected before trie expansion and retained on every orbit state",
            "orbit_orientation": "left ordinary end -> centre and right ordinary beginning -> exterior",
            "left_paths": len(left_paths),
            "right_paths": len(right_paths),
            "left_trie_nodes": len(left_trie.nodes),
            "right_trie_nodes": len(right_trie.nodes),
        },
        "stats": {
            "expanded_product_states": product["expanded_states"],
            "matched_orbit_transitions": product["matched_orbit_transitions"],
            "rejected_orbit_transitions": product["rejected_orbit_transitions"],
            "max_orbit_depth": product["max_orbit_depth"],
            "budget_exhausted": product["budget_exhausted"],
            "exact_closures_before_novelty": len(product["closure_pairs"]),
            "exact": len(exact),
            "mechanically_admitted": len(admitted),
            "mechanical_rejections": mechanical_rejections,
            "duplicate_rejections": duplicate_rejections,
            "intact_controls": len(controls),
            "longest_rendered_letters": max((row["audit"]["letters"] for row in all_rendered), default=0),
            "shortest_rendered_letters": min((row["audit"]["letters"] for row in all_rendered), default=0),
        },
        "rendered_candidates": rows,
        "rendered_controls": controls,
        "all_rendered_rows": all_rendered,
        "independent_audits": [
            "live two-sided character-orbit product",
            "independent normalized outside-in two-pointer scan",
            "independent forward/reverse SHA-256 comparison",
        ],
        "novelty_preflight": preflight,
        "grammar_exhaustion": {
            "status": "budget_exhausted" if product["budget_exhausted"] else "exhausted",
            "reason": (
                "the product exhausted all shared character frontiers before both complete semantic paths reached a "
                "39--60-letter terminal pair"
                if not product["budget_exhausted"] else
                "the bounded product reached its state budget before terminal closure"
            ),
            "dead_states_by_orbit": product["dead_states_by_orbit"],
            "dead_frontier_samples": product["dead_frontier_samples"],
        },
        "next_discriminator": {
            "operator": "add one held-out complete story frame with a distinct semantic role, then rerun the same centre-out product",
            "selection_rule": "choose the frame whose first dead frontier has the largest boundary-role support; do not edit a rendered tape",
            "falsifier": "if the added frame produces no new shared character frontier at the first exhausted orbit, reject it and keep the grammar closed",
        },
        "reader_gate": "closed; no exact candidate is human readability evidence until blinded readers compare intact and shuffled controls",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / f"{EXPERIMENT_ID}.json")
    parser.add_argument("--max-paths", type=int, default=DEFAULT_MAX_PATHS)
    parser.add_argument("--max-states", type=int, default=DEFAULT_MAX_STATES)
    args = parser.parse_args()
    result = run(max_paths=args.max_paths, max_states=args.max_states)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["stats"], sort_keys=True))
    for row in result["all_rendered_rows"][:4]:
        print(row["rendered"])


if __name__ == "__main__":
    main()
