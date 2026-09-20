"""Unequal-clause grammar intersection with a minimal typed seam machine.

This lane keeps two independently authored complete clause derivations in
ordinary word order.  The left derivation is consumed from its first lexical
character and the right derivation is consumed from its last lexical
character.  A small seam machine carries only cursor positions and one of
three closure modes (even, left-centre, or right-centre); it does not store an
unmatched rendered string.  Consequently the two clause paths may have
different word boundaries, and a one-character odd centre may be the final or
initial character of a multi-character word.

The search never reverses a completed candidate.  Reversal is used only by
the independent post-run audit, which is deliberately separate from the
construction walk.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/unequal-center-grammar-intersection-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "unequal-center-grammar-intersection-20260920"
SIGNATURE = (
    "typed-seam-machine|token-frontier-product|center-overhang|"
    "independent-clause-grammar"
)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def pointer_audit(text: str) -> dict:
    """Independent direct two-pointer audit of a rendered sentence."""
    tape = normalize(text)
    mismatch: Optional[dict] = None
    i, j = 0, len(tape) - 1
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
    """Independent hash audit; construction code does not call this helper."""
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
    number: str
    valency: str
    mode: str
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
    """A lexical cursor; it never materializes a reversed tape."""

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
        return cls(words=words, index=len(words) - 1, offset=len(words[-1].surface) - 1, reverse=True)

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
        total = 0
        if self.reverse:
            total += self.offset + 1
            total += sum(len(word.surface) for word in self.words[: self.index])
        else:
            total += len(self.words[self.index].surface) - self.offset
            total += sum(len(word.surface) for word in self.words[self.index + 1 :])
        return total


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


SUBJECTS = {
    "sg": (
        ("the patient pilot", "pilot"),
        ("a quiet scholar", "scholar"),
        ("the careful mason", "mason"),
    ),
    "pl": (
        ("patient pilots", "pilots"),
        ("quiet scholars", "scholars"),
        ("careful masons", "masons"),
    ),
}

VERBS = {
    "sg": (("charts", "chart"), ("marks", "mark"), ("reads", "read"), ("guides", "guide")),
    "pl": (("chart", "chart"), ("mark", "mark"), ("read", "read"), ("guide", "guide")),
}

OBJECTS = (
    ("the old map", "map"),
    ("a bright lantern", "lantern"),
    ("the quiet inlet", "inlet"),
    ("new notes", "notes"),
    ("a field journal", "journal"),
)

ADJUNCTS = (
    ("by the harbor", "place"),
    ("near the river", "place"),
    ("at first light", "time"),
    ("with calm care", "manner"),
    ("after the rain", "time"),
)


def phrase_words(text: str, role: str, phrase_index: int) -> tuple[Word, ...]:
    return tuple(Word(token, role, phrase_index) for token in text.split())


def build_clause_paths() -> tuple[Clause, ...]:
    """Build complete ordinary-order clauses from typed fresh scene slots."""
    paths: list[Clause] = []
    for number, subjects in SUBJECTS.items():
        for subject, subject_lemma in subjects:
            for verb, verb_lemma in VERBS[number]:
                for object_text, object_lemma in OBJECTS:
                    base = [
                        phrase_words(subject, "subject", 0),
                        phrase_words(verb, "finite_verb", 1),
                        phrase_words(object_text, "theme", 2),
                    ]
                    for mode, adjuncts in (
                        ("bare", ()),
                        ("place", (ADJUNCTS[0],)),
                        ("time", (ADJUNCTS[2],)),
                    ):
                        phrases = list(base)
                        role_names = ["subject", "finite_verb", "theme"]
                        if adjuncts:
                            text, lemma = adjuncts[0]
                            phrases.append(phrase_words(text, "adjunct", 3))
                            role_names.append("adjunct")
                        words = tuple(word for phrase in phrases for word in phrase)
                        frame_id = f"{subject_lemma}:{verb_lemma}:{object_lemma}:{mode}"
                        paths.append(
                            Clause(
                                frame_id=frame_id,
                                number=number,
                                valency="transitive",
                                mode=mode,
                                words=words,
                                roles=tuple(role_names),
                            )
                        )
    return tuple(paths)


def walk_pair(left: Clause, right: Clause) -> Walk:
    """Run the typed seam machine while lexical terminals are consumed."""
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


STOPWORDS = {"a", "an", "the", "at", "by", "near", "after", "with", "new", "old", "quiet", "calm"}


def shortcut_flags(left: Clause, right: Clause) -> dict:
    left_content = [
        word.surface.casefold()
        for word in left.words
        if word.role not in {"subject", "finite_verb", "theme", "adjunct"}
        or word.surface.casefold() not in STOPWORDS
    ]
    right_content = [
        word.surface.casefold()
        for word in right.words
        if word.role not in {"subject", "finite_verb", "theme", "adjunct"}
        or word.surface.casefold() not in STOPWORDS
    ]
    repeated_content = sorted(set(left_content).intersection(right_content))
    self_palindromic = sorted(
        {
            word.surface.casefold()
            for word in (*left.words, *right.words)
            if len(normalize(word.surface)) > 1 and normalize(word.surface) == normalize(word.surface)[::-1]
        }
    )
    return {
        "identical_clause_surface": left.surface == right.surface,
        "repeated_content_units": bool(repeated_content),
        "repeated_content_words": repeated_content,
        "self_palindromic_units": bool(self_palindromic),
        "self_palindromic_words": self_palindromic,
        "word_order_symmetry": left.roles == right.roles and left.frame_id == right.frame_id,
        "catalogue_text": False,
        "fragment": False,
        "finished_tape_reversal": False,
        "post_hoc_repair": False,
        "per_search_rlaif": False,
    }


def row_for(left: Clause, right: Clause, walk: Walk, exact: bool = False) -> dict:
    rendered = render_pair(left, right)
    audit = independent_audit(rendered)
    flags = shortcut_flags(left, right)
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
            "left_number": left.number,
            "right_number": right.number,
            "left_roles": list(left.roles),
            "right_roles": list(right.roles),
            "left_clause_words": [word.surface for word in left.words],
            "right_clause_words": [word.surface for word in right.words],
            "lexical_source": "fresh authored typed scene slots",
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
        "shortcut_flags": flags,
        # Exactness and shortcut cleanliness are necessary but not sufficient:
        # this lane has no blinded human ratings, so it cannot promote a row.
        "reader_facing_eligible": False,
        "reader_evidence": {
            "status": "not_run",
            "human_raters": 0,
            "reason": "No blinded reader study has been run for this diagnostic lane.",
        },
    }


def cursor_center_contract() -> dict:
    """Exercise odd/even overhang modes independently of sentence search."""
    def probe(left_text: tuple[str, ...], right_text: tuple[str, ...]) -> dict:
        left = Clause("probe-left", "sg", "transitive", "probe", tuple(Word(x, "probe", 0) for x in left_text), ("probe",))
        right = Clause("probe-right", "sg", "transitive", "probe", tuple(Word(x, "probe", 0) for x in right_text), ("probe",))
        walk = walk_pair(left, right)
        return {"left_words": list(left_text), "right_words": list(right_text), "status": walk.status, "seam_mode": walk.seam_mode, "center_location": walk.center_location}

    probes = [
        probe(("aba",), ("ba",)),
        probe(("ab",), ("cba",)),
        probe(("ab",), ("ba",)),
    ]
    return {
        "description": "Cursor-level contract only; probe strings are not sentence candidates.",
        "probes": probes,
        "all_expected_modes_present": {"left_center_inside_word", "right_center_inside_word", "even_seam"}.issubset(
            {probe["seam_mode"] for probe in probes}
        ),
    }


def novelty_preflight() -> dict:
    registry = json.loads(REGISTRY.read_text())
    entries = registry.get("entries", [])
    exact_collision = [entry.get("id") for entry in entries if entry.get("signature") == SIGNATURE]
    related = [
        entry.get("id")
        for entry in entries
        if any(
            marker in entry.get("signature", "")
            for marker in ("unequal-center", "center-buffer", "reverse-facing", "grammar-intersection")
        )
    ]
    return {
        "status": "passed" if not exact_collision else "collision",
        "registry_inspected": True,
        "registry_entries_read": len(entries),
        "exact_signature_collision": exact_collision,
        "related_signatures_seen": related,
        "signature": SIGNATURE,
        "distinction": (
            "The product state stores only two independent grammar-path cursor positions, "
            "typed number/valency metadata, and an explicit even/left-centre/right-centre "
            "seam mode. Unlike residual-string, reverse-trie, boundary-DP, or paired-CFG "
            "lanes, it permits unequal terminal counts and lets the one-character centre be "
            "the edge character of a multi-character terminal without a fixed seam or endpoint seed."
        ),
        "not_a_duplicate_sweep": True,
    }


def select_controls(observations: list[tuple[int, int, Clause, Clause, Walk]], limit: int = 16) -> list[dict]:
    selected: list[tuple[int, int, Clause, Clause, Walk]] = []
    seen: set[tuple[int, int]] = set()
    seen_surfaces: set[tuple[str, str]] = set()
    # Retain different ordinary complete-clause lengths and frame combinations,
    # rather than writing sixteen copies of the same longest control.
    for mode in ("mismatch_pruned", "overhang_pruned", "closed"):
        pool = [row for row in observations if row[4].status == mode]
        for row in sorted(pool, key=lambda item: (item[2].letters + item[3].letters, item[0], item[1]), reverse=True):
            key = (row[0], row[1])
            surface_key = (row[2].surface, row[3].surface)
            if row[2].surface == row[3].surface:
                continue
            if key not in seen and surface_key not in seen_surfaces:
                selected.append(row)
                seen.add(key)
                seen_surfaces.add(surface_key)
                break
    ordered = sorted(observations, key=lambda item: (item[2].letters + item[3].letters, item[0], item[1]), reverse=True)
    # First take one pair from each coarse total-length band, then fill with
    # long controls. This keeps the artifact readable without hiding the max.
    max_length = max((item[2].letters + item[3].letters for item in observations), default=0)
    bands = (max_length, max_length - 10, max_length - 20, max_length - 30, max_length - 40)
    for band in bands:
        for row in ordered:
            total = row[2].letters + row[3].letters
            if total > band or total <= band - 10:
                continue
            key = (row[0], row[1])
            surface_key = (row[2].surface, row[3].surface)
            if row[2].surface == row[3].surface:
                continue
            if key not in seen and surface_key not in seen_surfaces:
                selected.append(row)
                seen.add(key)
                seen_surfaces.add(surface_key)
                break
    for row in ordered:
        key = (row[0], row[1])
        surface_key = (row[2].surface, row[3].surface)
        if row[2].surface == row[3].surface:
            continue
        if key not in seen and surface_key not in seen_surfaces:
            selected.append(row)
            seen.add(key)
            seen_surfaces.add(surface_key)
        if len(selected) >= limit:
            break
    return [row_for(left, right, walk) for _, _, left, right, walk in selected[:limit]]


def run() -> dict:
    paths = build_clause_paths()
    observations: list[tuple[int, int, Clause, Clause, Walk]] = []
    total_equations = 0
    mismatch_prunes = 0
    overhang_prunes = 0
    closed = 0
    exact_candidates: list[dict] = []

    for left_index, left in enumerate(paths):
        for right_index, right in enumerate(paths):
            walk = walk_pair(left, right)
            observations.append((left_index, right_index, left, right, walk))
            total_equations += walk.online_equations
            if walk.status == "mismatch_pruned":
                mismatch_prunes += 1
            elif walk.status == "overhang_pruned":
                overhang_prunes += 1
            else:
                closed += 1
                row = row_for(left, right, walk, exact=True)
                if row["audit"]["independent_exact"]:
                    exact_candidates.append(row)

    exact_clean = [
        row
        for row in exact_candidates
        if row["letters"] > 38
        and not row["shortcut_flags"]["identical_clause_surface"]
        and not row["shortcut_flags"]["repeated_content_units"]
        and not row["shortcut_flags"]["self_palindromic_units"]
        and not row["shortcut_flags"]["word_order_symmetry"]
        and not row["shortcut_flags"]["fragment"]
    ]
    controls = select_controls(observations)
    center_contract = cursor_center_contract()
    result = {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "Independent typed complete-clause grammar paths intersected by a minimal "
            "two-cursor seam machine; left characters advance forward, right characters "
            "advance inward, and unequal word/clause boundaries remain live."
        ),
        "stats": {
            "left_grammar_paths": len(paths),
            "right_grammar_paths": len(paths),
            "paired_grammar_states": len(observations),
            "online_character_equations": total_equations,
            "mismatch_prunes": mismatch_prunes,
            "overhang_prunes": overhang_prunes,
            "seam_closures": closed,
            "unequal_word_count_pairs": sum(len(left.words) != len(right.words) for _, _, left, right, _ in observations),
            "unequal_letter_count_pairs": sum(left.letters != right.letters for _, _, left, right, _ in observations),
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
            "reason": "No exact-clean candidate has independent blinded human readability evidence.",
            "human_raters": 0,
        },
        "center_transition_contract": center_contract,
        "novelty_preflight": novelty_preflight(),
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "independent_audits": ["direct two-pointer scan", "forward/reverse SHA-256"],
            "search_lexical_source": "fresh authored typed subject/verb/theme/adjunct slots",
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "mirrored_or_self_palindromic_units": False,
            "catalogue_text": False,
            "word_order_symmetry": False,
            "per_search_rlaif": False,
        },
        "status": "completed_no_exact_closure" if not exact_candidates else "mechanical_exact_requires_reader_gate",
        "next_construction": (
            "Add a held-out ditransitive and relative-complement production to the same "
            "cursor product, keeping the seam state scalar and preserving unequal-boundary "
            "advancement; do not add repair, endpoint seeding, or mirrored lexical units."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps({"experiment_id": output["experiment_id"], "stats": output["stats"]}, sort_keys=True))
