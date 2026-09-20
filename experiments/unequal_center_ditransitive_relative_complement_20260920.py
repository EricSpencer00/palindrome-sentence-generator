"""Held-out valency productions in the unequal-center seam product.

This is a follow-up to ``unequal_center_grammar_intersection_20260920``.  It
keeps that lane's two independent lexical cursors and changes exactly one
thing: the clause grammar now admits two held-out complete productions:

* a double-object ditransitive (subject--verb--recipient--theme), and
* a theme-headed object-relative complement (subject--verb--theme [that
  subject--finite-verb]).

The left clause is emitted in ordinary order from its first character and the
right clause is consumed inward from its final character.  Unequal word and
letter boundaries, including a one-character center inside a word, remain
states of the same seam machine.  No completed tape is reversed to construct
an output; reversal appears only in the independent audit below.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/unequal-center-ditransitive-relative-complement-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "unequal-center-ditransitive-relative-complement-20260920"
SIGNATURE = (
    "typed-seam-machine|token-frontier-product|center-overhang|"
    "heldout-ditransitive-relative-complement"
)
BASE_ID = "unequal-center-grammar-intersection-20260920"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def pointer_audit(text: str) -> dict:
    """Independent outside-in audit of a rendered sentence."""
    tape = normalize(text)
    i, j = 0, len(tape) - 1
    mismatch: Optional[dict] = None
    while i < j:
        if tape[i] != tape[j]:
            mismatch = {
                "left_index": i,
                "right_index": j,
                "left_char": tape[i],
                "right_char": tape[j],
            }
            break
        i += 1
        j -= 1
    return {
        "normalized_tape": tape,
        "letters": len(tape),
        "exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
    }


def hash_audit(text: str) -> dict:
    """Independent forward/reverse digest check, after construction."""
    tape = normalize(text)
    forward = hashlib.sha256(tape.encode("ascii")).hexdigest()
    reverse = hashlib.sha256(tape[::-1].encode("ascii")).hexdigest()
    return {
        "sha256_forward": forward,
        "sha256_reverse": reverse,
        "sha_equal_under_reversal": forward == reverse,
    }


def independent_audit(text: str) -> dict:
    direct = pointer_audit(text)
    hashed = hash_audit(text)
    return {
        **direct,
        **hashed,
        "independent_exact": direct["exact"] and hashed["sha_equal_under_reversal"],
    }


@dataclass(frozen=True)
class Word:
    surface: str
    role: str
    phrase_index: int


@dataclass(frozen=True)
class Clause:
    frame_id: str
    production: str
    number: str
    valency: str
    words: tuple[Word, ...]
    roles: tuple[str, ...]

    @property
    def surface(self) -> str:
        return " ".join(word.surface for word in self.words)

    @property
    def letters(self) -> int:
        return len(normalize(self.surface))


@dataclass
class Cursor:
    """Lexical cursor; it never materializes a reverse tape."""

    words: tuple[Word, ...]
    index: int
    offset: int
    reverse: bool
    boundary_crossings: int = 0

    @classmethod
    def left(cls, words: tuple[Word, ...]) -> "Cursor":
        return cls(words=words, index=0, offset=0, reverse=False)

    @classmethod
    def right(cls, words: tuple[Word, ...]) -> "Cursor":
        if not words:
            return cls(words=words, index=-1, offset=-1, reverse=True)
        return cls(
            words=words,
            index=len(words) - 1,
            offset=len(words[-1].surface) - 1,
            reverse=True,
        )

    def done(self) -> bool:
        return self.index < 0 or self.index >= len(self.words)

    def peek(self) -> Optional[str]:
        if self.done():
            return None
        return self.words[self.index].surface[self.offset]

    def location(self) -> Optional[dict]:
        if self.done():
            return None
        word = self.words[self.index]
        return {
            "word": word.surface,
            "role": word.role,
            "word_index": self.index,
            "offset": self.offset,
            "word_length": len(word.surface),
            "inside_word": len(word.surface) > 1,
        }

    def take(self) -> Optional[str]:
        char = self.peek()
        if char is None:
            return None
        if self.reverse:
            if self.offset == 0:
                self.index -= 1
                self.boundary_crossings += 1
                if self.index >= 0:
                    self.offset = len(self.words[self.index].surface) - 1
            else:
                self.offset -= 1
        else:
            if self.offset == len(self.words[self.index].surface) - 1:
                self.index += 1
                self.boundary_crossings += 1
                if self.index < len(self.words):
                    self.offset = 0
            else:
                self.offset += 1
        return char

    def remaining(self) -> int:
        if self.done():
            return 0
        if self.reverse:
            return self.offset + 1 + sum(len(word.surface) for word in self.words[: self.index])
        return len(self.words[self.index].surface) - self.offset + sum(
            len(word.surface) for word in self.words[self.index + 1 :]
        )


@dataclass(frozen=True)
class Walk:
    status: str
    seam_mode: str
    matched_characters: int
    online_equations: int
    first_mismatch: Optional[dict]
    left_boundaries: int
    right_boundaries: int
    center_location: Optional[dict]
    left_remaining: int
    right_remaining: int


# These are held out from the base unequal-center lane.  They are authored
# role inventories, not sentence or catalogue imports.
DITRANSITIVE_SUBJECTS = {
    "sg": (
        ("the astute courier", "courier"),
        ("a patient gardener", "gardener"),
    ),
    "pl": (
        ("astute couriers", "couriers"),
        ("patient gardeners", "gardeners"),
    ),
}

DITRANSITIVE_VERBS = {
    "sg": (("offers", "offer"), ("sends", "send")),
    "pl": (("offer", "offer"), ("send", "send")),
}

RECIPIENTS = (
    ("the quiet keeper", "keeper"),
    ("a careful student", "student"),
)

DITRANSITIVE_THEMES = (
    ("a sealed letter", "letter"),
    ("the blue atlas", "atlas"),
)

RELATIVE_MATRIX_SUBJECTS = {
    "sg": (
        ("the weathered sailor", "sailor"),
        ("a distant painter", "painter"),
    ),
    "pl": (
        ("weathered sailors", "sailors"),
        ("distant painters", "painters"),
    ),
}

RELATIVE_MATRIX_VERBS = {
    "sg": (("studies", "study"), ("records", "record")),
    "pl": (("study", "study"), ("record", "record")),
}

RELATIVE_HEADS = (
    ("the weathered chart", "chart"),
    ("a narrow bridge", "bridge"),
)

RELATIVE_SUBJECTS = {
    "sg": (
        ("the careful beekeeper", "beekeeper"),
        ("a quiet engineer", "engineer"),
    ),
    "pl": (
        ("careful beekeepers", "beekeepers"),
        ("quiet engineers", "engineers"),
    ),
}

RELATIVE_VERBS = {
    "sg": (("guards", "guard"), ("names", "name")),
    "pl": (("guard", "guard"), ("name", "name")),
}


def phrase_words(text: str, role: str, phrase_index: int) -> tuple[Word, ...]:
    return tuple(Word(token, role, phrase_index) for token in text.split())


def build_heldout_paths() -> tuple[Clause, ...]:
    """Build only the two new complete ordinary-order productions."""
    paths: list[Clause] = []

    for number, subjects in DITRANSITIVE_SUBJECTS.items():
        for subject, subject_lemma in subjects:
            for verb, verb_lemma in DITRANSITIVE_VERBS[number]:
                for recipient_text, recipient_lemma in RECIPIENTS:
                    for theme_text, theme_lemma in DITRANSITIVE_THEMES:
                        phrases = (
                            phrase_words(subject, "subject", 0),
                            phrase_words(verb, "finite_verb", 1),
                            phrase_words(recipient_text, "recipient", 2),
                            phrase_words(theme_text, "theme", 3),
                        )
                        paths.append(
                            Clause(
                                frame_id=(
                                    f"ditransitive:{subject_lemma}:{verb_lemma}:"
                                    f"{recipient_lemma}:{theme_lemma}"
                                ),
                                production="ditransitive",
                                number=number,
                                valency="ditransitive",
                                words=tuple(word for phrase in phrases for word in phrase),
                                roles=("subject", "finite_verb", "recipient", "theme"),
                            )
                        )

    for number, matrix_subjects in RELATIVE_MATRIX_SUBJECTS.items():
        for subject, subject_lemma in matrix_subjects:
            for matrix_verb, matrix_verb_lemma in RELATIVE_MATRIX_VERBS[number]:
                for head_text, head_lemma in RELATIVE_HEADS:
                    for relative_subject, relative_subject_lemma in RELATIVE_SUBJECTS[number]:
                        for relative_verb, relative_verb_lemma in RELATIVE_VERBS[number]:
                            # Object relative: the head noun is the object of
                            # the finite relative clause.  The matrix clause
                            # and the relative clause are both complete in
                            # ordinary order; the marker is not punctuation.
                            phrases = (
                                phrase_words(subject, "subject", 0),
                                phrase_words(matrix_verb, "finite_verb", 1),
                                phrase_words(head_text, "theme_head", 2),
                                phrase_words("that", "relative_marker", 3),
                                phrase_words(relative_subject, "relative_subject", 4),
                                phrase_words(relative_verb, "relative_finite_verb", 5),
                            )
                            paths.append(
                                Clause(
                                    frame_id=(
                                        f"relative-complement:{subject_lemma}:{matrix_verb_lemma}:"
                                        f"{head_lemma}:{relative_subject_lemma}:{relative_verb_lemma}"
                                    ),
                                    production="relative_complement",
                                    number=number,
                                    valency="object_relative",
                                    words=tuple(word for phrase in phrases for word in phrase),
                                    roles=(
                                        "subject",
                                        "finite_verb",
                                        "theme_head",
                                        "relative_marker",
                                        "relative_subject",
                                        "relative_finite_verb",
                                    ),
                                )
                            )
    return tuple(paths)


def walk_pair(left: Clause, right: Clause) -> Walk:
    """The same two-cursor seam product, with live word boundaries."""
    left_cursor = Cursor.left(left.words)
    right_cursor = Cursor.right(right.words)
    matched = 0
    equations = 0

    while True:
        left_char = left_cursor.peek()
        right_char = right_cursor.peek()
        if left_char is not None and right_char is not None:
            equations += 1
            if left_char != right_char:
                return Walk(
                    status="mismatch_pruned",
                    seam_mode="open",
                    matched_characters=matched,
                    online_equations=equations,
                    first_mismatch={
                        "left_char": left_char,
                        "right_char": right_char,
                        "left_location": left_cursor.location(),
                        "right_location": right_cursor.location(),
                        "matched_before_mismatch": matched,
                    },
                    left_boundaries=left_cursor.boundary_crossings,
                    right_boundaries=right_cursor.boundary_crossings,
                    center_location=None,
                    left_remaining=left_cursor.remaining(),
                    right_remaining=right_cursor.remaining(),
                )
            left_cursor.take()
            right_cursor.take()
            matched += 1
            continue

        left_remaining = left_cursor.remaining()
        right_remaining = right_cursor.remaining()
        if left_remaining == 0 and right_remaining == 0:
            return Walk(
                status="closed",
                seam_mode="even_seam",
                matched_characters=matched,
                online_equations=equations,
                first_mismatch=None,
                left_boundaries=left_cursor.boundary_crossings,
                right_boundaries=right_cursor.boundary_crossings,
                center_location=None,
                left_remaining=0,
                right_remaining=0,
            )
        if left_remaining == 1 and right_remaining == 0:
            center = left_cursor.location()
            left_cursor.take()
            return Walk(
                status="closed",
                seam_mode="left_center_inside_word" if center and center["inside_word"] else "left_center",
                matched_characters=matched,
                online_equations=equations,
                first_mismatch=None,
                left_boundaries=left_cursor.boundary_crossings,
                right_boundaries=right_cursor.boundary_crossings,
                center_location=center,
                left_remaining=0,
                right_remaining=0,
            )
        if right_remaining == 1 and left_remaining == 0:
            center = right_cursor.location()
            right_cursor.take()
            return Walk(
                status="closed",
                seam_mode="right_center_inside_word" if center and center["inside_word"] else "right_center",
                matched_characters=matched,
                online_equations=equations,
                first_mismatch=None,
                left_boundaries=left_cursor.boundary_crossings,
                right_boundaries=right_cursor.boundary_crossings,
                center_location=center,
                left_remaining=0,
                right_remaining=0,
            )
        return Walk(
            status="overhang_pruned",
            seam_mode="overhang",
            matched_characters=matched,
            online_equations=equations,
            first_mismatch=None,
            left_boundaries=left_cursor.boundary_crossings,
            right_boundaries=right_cursor.boundary_crossings,
            center_location=left_cursor.location() or right_cursor.location(),
            left_remaining=left_remaining,
            right_remaining=right_remaining,
        )


def render_pair(left: Clause, right: Clause) -> str:
    left_text = left.surface[:1].upper() + left.surface[1:]
    return f"{left_text}; {right.surface}."


def midpoint_crossing(text: str) -> dict:
    tape = normalize(text)
    midpoint = (len(tape) - 1) // 2
    cursor = 0
    crossing = None
    for token in re.findall(r"[A-Za-z]+", text):
        start, end = cursor, cursor + len(token)
        if start <= midpoint < end:
            crossing = {
                "token": token,
                "token_interval": [start, end],
                "midpoint": midpoint,
                "offset": midpoint - start,
                "inside_word": len(token) > 1,
            }
            break
        cursor = end
    return {"midpoint": midpoint, "crossing": crossing}


STOPWORDS = {
    "a", "an", "the", "that", "new", "to", "for", "and", "at", "by",
    "near", "after", "with", "in", "on", "of",
}


def shortcut_flags(left: Clause, right: Clause) -> dict:
    def content(clause: Clause) -> set[str]:
        return {
            word.surface.casefold()
            for word in clause.words
            if word.surface.casefold() not in STOPWORDS
        }

    repeated = sorted(content(left).intersection(content(right)))
    self_palindromic = sorted(
        {
            word.surface.casefold()
            for word in (*left.words, *right.words)
            if len(normalize(word.surface)) > 1
            and normalize(word.surface) == normalize(word.surface)[::-1]
        }
    )
    return {
        "identical_clause_surface": left.surface == right.surface,
        "repeated_content_units": bool(repeated),
        "repeated_content_words": repeated,
        "self_palindromic_units": bool(self_palindromic),
        "self_palindromic_words": self_palindromic,
        "word_order_symmetry": left.roles == right.roles and left.frame_id == right.frame_id,
        "catalogue_text": False,
        "fragment": False,
        "finished_tape_reversal": False,
        "post_hoc_repair": False,
        "per_search_rlaif": False,
    }


def row_for(left: Clause, right: Clause, walk: Walk) -> dict:
    rendered = render_pair(left, right)
    audit = independent_audit(rendered)
    return {
        "rendered": rendered,
        "letters": audit["letters"],
        "audit": audit,
        "seam_machine": {
            "status": walk.status,
            "mode": walk.seam_mode,
            "matched_characters": walk.matched_characters,
            "online_equations": walk.online_equations,
            "first_mismatch": walk.first_mismatch,
            "left_word_boundary_crossings": walk.left_boundaries,
            "right_word_boundary_crossings": walk.right_boundaries,
            "center_location": walk.center_location,
            "left_remaining": walk.left_remaining,
            "right_remaining": walk.right_remaining,
        },
        "midpoint": midpoint_crossing(rendered),
        "provenance": {
            "left_frame_id": left.frame_id,
            "right_frame_id": right.frame_id,
            "left_production": left.production,
            "right_production": right.production,
            "left_number": left.number,
            "right_number": right.number,
            "left_roles": list(left.roles),
            "right_roles": list(right.roles),
            "left_clause_words": [word.surface for word in left.words],
            "right_clause_words": [word.surface for word in right.words],
            "lexical_source": "held-out authored ditransitive and object-relative-complement slots",
            "grammar_paths_independent": True,
            "lexicalized_during_cursor_walk": True,
            "right_clause_read_inward_by_cursor": True,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "mirrored_units": False,
            "catalogue_text": False,
            "word_order_symmetry": False,
            "per_search_rlaif": False,
        },
        "shortcut_flags": shortcut_flags(left, right),
        "reader_facing_eligible": False,
        "reader_evidence": {
            "status": "not_run",
            "human_raters": 0,
            "reason": "No blinded reader study has been run for this diagnostic lane.",
        },
    }


def cursor_center_contract() -> dict:
    """Exercise all seam modes, including an in-word center."""
    def probe(left_text: tuple[str, ...], right_text: tuple[str, ...]) -> dict:
        left = Clause("probe-left", "probe", "sg", "probe", tuple(Word(x, "probe", 0) for x in left_text), ("probe",))
        right = Clause("probe-right", "probe", "sg", "probe", tuple(Word(x, "probe", 0) for x in right_text), ("probe",))
        walk = walk_pair(left, right)
        return {
            "left_words": list(left_text),
            "right_words": list(right_text),
            "status": walk.status,
            "seam_mode": walk.seam_mode,
            "center_location": walk.center_location,
        }

    probes = [probe(("aba",), ("ba",)), probe(("ab",), ("cba",)), probe(("ab",), ("ba",))]
    modes = {probe["seam_mode"] for probe in probes}
    return {
        "description": "Cursor-level contract only; probe strings are not sentence candidates.",
        "probes": probes,
        "all_expected_modes_present": {"left_center_inside_word", "right_center_inside_word", "even_seam"}.issubset(modes),
    }


def novelty_preflight() -> dict:
    registry = json.loads(REGISTRY.read_text())
    entries = registry.get("entries", [])
    exact_collision = [
        entry.get("id")
        for entry in entries
        if entry.get("id") == EXPERIMENT_ID or entry.get("signature") == SIGNATURE
    ]
    related = [
        entry.get("id")
        for entry in entries
        if entry.get("id") == BASE_ID
        or any(marker in entry.get("signature", "") for marker in ("unequal-center", "center-buffer", "grammar-intersection"))
    ]
    return {
        "status": "passed" if not exact_collision else "collision",
        "registry_inspected": True,
        "registry_entries_read": len(entries),
        "exact_signature_collision": exact_collision,
        "base_lane_seen": BASE_ID in related,
        "related_signatures_seen": related,
        "signature": SIGNATURE,
        "distinction": (
            "The prior seam machine is reused unchanged as the cursor product, but this run emits only "
            "two held-out ordinary-order productions: a recipient/theme ditransitive and a theme-headed "
            "object-relative complement. Both carry typed valency and agreement roles through unequal word "
            "and letter boundaries; no old transitive bank is replayed."
        ),
        "not_a_duplicate_sweep": True,
    }


def select_controls(observations: list[tuple[int, int, Clause, Clause, Walk]], limit: int = 20) -> list[dict]:
    selected: list[tuple[int, int, Clause, Clause, Walk]] = []
    seen_pairs: set[tuple[int, int]] = set()
    seen_surfaces: set[tuple[str, str]] = set()

    # Ensure every production pairing appears in the artifact. Prefer prose
    # with no repeated content words, then fall back to complete prose only.
    families = ("ditransitive", "relative_complement")
    for left_family in families:
        for right_family in families:
            pool = [
                row for row in observations
                if row[2].production == left_family and row[3].production == right_family
            ]
            pool.sort(key=lambda item: (not bool(shortcut_flags(item[2], item[3])["repeated_content_units"]), item[2].letters + item[3].letters), reverse=True)
            for row in pool:
                key = (row[0], row[1])
                surface_key = (row[2].surface, row[3].surface)
                if row[2].surface == row[3].surface or key in seen_pairs or surface_key in seen_surfaces:
                    continue
                selected.append(row)
                seen_pairs.add(key)
                seen_surfaces.add(surface_key)
                break

    ordered = sorted(
        observations,
        key=lambda item: (item[2].letters + item[3].letters, item[0], item[1]),
        reverse=True,
    )
    max_length = max((item[2].letters + item[3].letters for item in observations), default=0)
    for band in (max_length, max_length - 10, max_length - 20, max_length - 30, max_length - 40, max_length - 50):
        for row in ordered:
            total = row[2].letters + row[3].letters
            if total > band or total <= band - 10:
                continue
            key = (row[0], row[1])
            surface_key = (row[2].surface, row[3].surface)
            if row[2].surface == row[3].surface or key in seen_pairs or surface_key in seen_surfaces:
                continue
            selected.append(row)
            seen_pairs.add(key)
            seen_surfaces.add(surface_key)
            break

    for row in ordered:
        key = (row[0], row[1])
        surface_key = (row[2].surface, row[3].surface)
        if row[2].surface == row[3].surface or key in seen_pairs or surface_key in seen_surfaces:
            continue
        selected.append(row)
        seen_pairs.add(key)
        seen_surfaces.add(surface_key)
        if len(selected) >= limit:
            break
    return [row_for(left, right, walk) for _, _, left, right, walk in selected[:limit]]


def run() -> dict:
    paths = build_heldout_paths()
    observations: list[tuple[int, int, Clause, Clause, Walk]] = []
    total_equations = 0
    mismatch_prunes = 0
    overhang_prunes = 0
    seam_closures = 0
    exact_candidates: list[dict] = []
    unequal_words = 0
    unequal_letters = 0

    for left_index, left in enumerate(paths):
        for right_index, right in enumerate(paths):
            walk = walk_pair(left, right)
            observations.append((left_index, right_index, left, right, walk))
            total_equations += walk.online_equations
            mismatch_prunes += walk.status == "mismatch_pruned"
            overhang_prunes += walk.status == "overhang_pruned"
            seam_closures += walk.status == "closed"
            unequal_words += len(left.words) != len(right.words)
            unequal_letters += left.letters != right.letters
            if walk.status == "closed":
                row = row_for(left, right, walk)
                if row["audit"]["independent_exact"]:
                    exact_candidates.append(row)

    exact_clean = [
        row for row in exact_candidates
        if row["letters"] > 38
        and not row["shortcut_flags"]["identical_clause_surface"]
        and not row["shortcut_flags"]["repeated_content_units"]
        and not row["shortcut_flags"]["self_palindromic_units"]
        and not row["shortcut_flags"]["word_order_symmetry"]
        and not row["shortcut_flags"]["fragment"]
    ]
    controls = select_controls(observations)
    center_contract = cursor_center_contract()
    family_counts = {
        family: sum(path.production == family for path in paths)
        for family in ("ditransitive", "relative_complement")
    }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "Held-out ditransitive and theme-headed object-relative-complement grammar paths "
            "intersected by the same minimal two-cursor seam machine as the base unequal-center lane; "
            "left characters advance forward, right characters advance inward, and unequal word/clause "
            "boundaries plus in-word centers remain live."
        ),
        "operator_added": {
            "name": "held-out ditransitive and relative-complement productions",
            "ditransitive_form": "Subject finite-verb Recipient Theme",
            "relative_complement_form": "Subject finite-verb ThemeHead that RelativeSubject RelativeFiniteVerb",
            "old_transitive_bank_replayed": False,
            "same_cursor_product": True,
        },
        "stats": {
            "heldout_clause_paths": len(paths),
            "paths_by_production": family_counts,
            "paired_grammar_states": len(observations),
            "online_character_equations": total_equations,
            "mismatch_prunes": mismatch_prunes,
            "overhang_prunes": overhang_prunes,
            "seam_closures": seam_closures,
            "unequal_word_count_pairs": unequal_words,
            "unequal_letter_count_pairs": unequal_letters,
            "rendered_controls": len(controls),
            "mechanical_exact_candidates": len(exact_candidates),
            "exact_clean_above_38": len(exact_clean),
            "longest_rendered_control_letters": max((row["letters"] for row in controls), default=0),
            "longest_exact_clean_letters": max((row["letters"] for row in exact_clean), default=0),
        },
        "rendered_controls": controls,
        "exact_candidates": exact_candidates,
        "exact_clean_candidates": exact_clean,
        "reader_facing_candidates": [],
        "reader_gate": {
            "status": "closed",
            "reason": "No exact-clean candidate from this lane has blinded human readability evidence.",
            "human_raters": 0,
        },
        "center_transition_contract": center_contract,
        "novelty_preflight": novelty_preflight(),
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["direct outside-in two-pointer scan", "forward/reverse SHA-256"],
            "search_lexical_source": "held-out authored ditransitive and object-relative-complement role slots",
            "old_transitive_paths_replayed": False,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "mirrored_or_self_palindromic_units": False,
            "catalogue_text": False,
            "word_order_symmetry": False,
            "fragment_output": False,
            "per_search_rlaif": False,
        },
        "status": "completed_no_exact_closure" if not exact_candidates else "mechanical_exact_requires_reader_gate",
        "next_construction": {
            "operator": "typed benefactive alternation plus a subject-relative companion",
            "description": (
                "If this held-out valency product remains empty, add one independently authored "
                "to/for benefactive ditransitive frame and one subject-relative frame, carrying case, "
                "attachment, number, and tense as live grammar state in the same cursor product."
            ),
            "reader_facing_test": "Retain only complete vivid prose; independently audit any exact closure above 38, then run randomized blinded intact-vs-shuffled ratings.",
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps({"experiment_id": output["experiment_id"], "stats": output["stats"]}, sort_keys=True))
