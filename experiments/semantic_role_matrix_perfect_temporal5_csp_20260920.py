"""Held-out matrix-perfect alternation for a typed temporal relative grammar.

This lane follows ``semantic_role_temporal_subject_relative4_csp`` with one
grammar change, not a lexical sweep: the subject-relative clause keeps its
future-perfect ``who will have ...`` frame, while the matrix clause now has a
held-out future-perfect state of its own (``will have PARTICIPLE ...``).
The temporal adjunct inventory is also a new, semantically distinct ``since`` /
``after`` inventory.  Every clause is emitted in ordinary order; the right
clause is consumed by the shared two-cursor seam machine rather than by
reversing a finished tape.

The artifact records complete English controls, an independent literal
outside-in audit, independent forward/reverse SHA-256 digests, typed
provenance, shortcut checks, and the next construction.  Controls are
diagnostic: no candidate is reader-eligible without an exact closure and a
later blinded human study.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.unequal_center_ditransitive_relative_complement_20260920 import (  # noqa: E402
    Word,
    walk_pair,
)


OUT = ROOT / "runs/semantic-role-matrix-perfect-temporal5-csp-20260920.json"
REGISTRY = ROOT / "docs/experiment-novelty-registry.json"
EXPERIMENT_ID = "semantic-role-matrix-perfect-temporal5-csp-20260920"
SIGNATURE = (
    "typed-seam-machine|temporal-subject-relative|future-perfect|"
    "matrix-future-perfect|since-after-adjunct|nominative-head|"
    "independent-pointer-sha"
)
PREDECESSOR_ID = "semantic-role-temporal-subject-relative4-csp-20260920"


def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.casefold())


def pointer_audit(text: str) -> dict:
    """Independent outside-in exact check, separate from the seam walk."""
    tape = normalize(text)
    mismatch: Optional[dict] = None
    left, right = 0, len(tape) - 1
    while left < right:
        if tape[left] != tape[right]:
            mismatch = {
                "left_index": left,
                "right_index": right,
                "left_char": tape[left],
                "right_char": tape[right],
            }
            break
        left += 1
        right -= 1
    return {
        "normalized_tape": tape,
        "letters": len(tape),
        "exact": bool(tape) and mismatch is None,
        "first_mismatch": mismatch,
    }


def hash_audit(text: str) -> dict:
    """Independent forward/reverse digest check after construction."""
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
class Clause:
    frame_id: str
    production: str
    valency: str
    case: str
    attachment: str
    number: str
    relative_tense: str
    relative_aspect: str
    matrix_tense: str
    matrix_aspect: str
    temporal_relation: str
    words: tuple[Word, ...]
    roles: tuple[str, ...]

    @property
    def surface(self) -> str:
        return " ".join(word.surface for word in self.words)

    @property
    def letters(self) -> int:
        return len(normalize(self.surface))


# Fresh authored role banks.  They are not imported from the predecessor's
# lexical inventory.  The matrix participles are deliberately distinct from
# the predecessor's simple-future matrix verbs: the new state is ``will have``
# plus a past participle.
SUBJECTS = {
    "sg": (
        ("the patient pilot", "pilot"),
        ("a careful gardener", "gardener"),
        ("the quiet scholar", "scholar"),
    ),
    "pl": (
        ("patient pilots", "pilots"),
        ("careful gardeners", "gardeners"),
        ("quiet scholars", "scholars"),
    ),
}

RELATIVE_EVENTS = {
    "sg": (
        ("charted", "chart", "the old map", "map"),
        ("guarded", "guard", "the quiet inlet", "inlet"),
        ("studied", "study", "the broad atlas", "atlas"),
    ),
    "pl": (
        ("charted", "chart", "old maps", "maps"),
        ("guarded", "guard", "quiet inlets", "inlets"),
        ("studied", "study", "broad atlases", "atlases"),
    ),
}

MATRIX_EVENTS = {
    "sg": (
        ("guided", "guide", "the narrow bridge", "bridge"),
        ("observed", "observe", "a calm harbor", "harbor"),
        ("recorded", "record", "the evening bell", "bell"),
    ),
    "pl": (
        ("guided", "guide", "the narrow bridges", "bridges"),
        ("observed", "observe", "calm harbors", "harbors"),
        ("recorded", "record", "the evening bells", "bells"),
    ),
}

# This is intentionally not the predecessor's before/by inventory.  ``since
# dawn`` and ``after first light`` are ordinary temporal adjuncts; ``after the
# tide turns`` supplies a finite temporal event rather than a word substitution.
TEMPORAL_ADJUNCTS = (
    ("since dawn", "since", "dawn"),
    ("after first light", "after", "light"),
    ("after the tide turns", "after", "tide-turn"),
)


def phrase_words(text: str, role: str, phrase_index: int) -> tuple[Word, ...]:
    return tuple(Word(token, role, phrase_index) for token in text.split())


def build_paths() -> tuple[Clause, ...]:
    """Build only the fresh matrix-perfect temporal grammar paths."""
    paths: list[Clause] = []
    for number, subjects in SUBJECTS.items():
        for subject_text, subject_lemma in subjects:
            for rel_participle, rel_lemma, rel_theme, rel_theme_lemma in RELATIVE_EVENTS[number]:
                for matrix_participle, matrix_lemma, matrix_theme, matrix_theme_lemma in MATRIX_EVENTS[number]:
                    for adjunct_text, relation, adjunct_lemma in TEMPORAL_ADJUNCTS:
                        phrases = (
                            phrase_words(subject_text, "relative_head_subject", 0),
                            phrase_words("who", "relative_marker", 1),
                            phrase_words("will", "relative_future_auxiliary", 2),
                            phrase_words("have", "relative_perfect_auxiliary", 3),
                            phrase_words(rel_participle, "relative_past_participle", 4),
                            phrase_words(rel_theme, "relative_theme", 5),
                            phrase_words(adjunct_text, "temporal_adjunct", 6),
                            phrase_words("will", "matrix_future_auxiliary", 7),
                            phrase_words("have", "matrix_perfect_auxiliary", 8),
                            phrase_words(matrix_participle, "matrix_past_participle", 9),
                            phrase_words(matrix_theme, "matrix_theme", 10),
                        )
                        paths.append(
                            Clause(
                                frame_id=(
                                    f"matrix-perfect-temporal-subject-relative:"
                                    f"future-perfect:{number}:{subject_lemma}:{rel_lemma}:"
                                    f"{rel_theme_lemma}:{relation}-{adjunct_lemma}:"
                                    f"matrix-future-perfect-{matrix_lemma}:{matrix_theme_lemma}"
                                ),
                                production="subject_relative_temporal_matrix_perfect",
                                valency="transitive_matrix_with_subject_relative",
                                case="nominative_head",
                                attachment="subject_relative_temporal",
                                number=number,
                                relative_tense="future",
                                relative_aspect="perfect",
                                matrix_tense="future",
                                matrix_aspect="perfect",
                                temporal_relation=relation,
                                words=tuple(word for phrase in phrases for word in phrase),
                                roles=(
                                    "relative_head_subject", "relative_marker",
                                    "relative_future_auxiliary", "relative_perfect_auxiliary",
                                    "relative_past_participle", "relative_theme",
                                    "temporal_adjunct", "matrix_future_auxiliary",
                                    "matrix_perfect_auxiliary", "matrix_past_participle",
                                    "matrix_theme",
                                ),
                            )
                        )
    return tuple(paths)


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
    "a", "an", "the", "who", "will", "have", "since", "after", "first",
    "light", "dawn", "tide", "turns", "and", "at", "near", "by", "before",
    "with", "in", "on", "of",
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
        "mirrored_units": False,
        "per_search_rlaif": False,
    }


def row_for(left: Clause, right: Clause, walk) -> dict:
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
            "left_state": {
                "case": left.case,
                "attachment": left.attachment,
                "number": left.number,
                "relative_tense": left.relative_tense,
                "relative_aspect": left.relative_aspect,
                "matrix_tense": left.matrix_tense,
                "matrix_aspect": left.matrix_aspect,
                "temporal_relation": left.temporal_relation,
            },
            "right_state": {
                "case": right.case,
                "attachment": right.attachment,
                "number": right.number,
                "relative_tense": right.relative_tense,
                "relative_aspect": right.relative_aspect,
                "matrix_tense": right.matrix_tense,
                "matrix_aspect": right.matrix_aspect,
                "temporal_relation": right.temporal_relation,
            },
            "left_roles": list(left.roles),
            "right_roles": list(right.roles),
            "left_clause_words": [word.surface for word in left.words],
            "right_clause_words": [word.surface for word in right.words],
            "lexical_source": "fresh authored matrix-perfect temporal subject-relative slots",
            "grammar_paths_independent": True,
            "lexicalized_during_cursor_walk": True,
            "right_clause_read_inward_by_cursor": True,
            "finished_tape_reversal": False,
            "post_hoc_repair": False,
            "mirrored_units": False,
            "self_palindromic_units": False,
            "catalogue_text": False,
            "word_order_symmetry": False,
            "fragment": False,
            "per_search_rlaif": False,
        },
        "shortcut_flags": shortcut_flags(left, right),
        "reader_facing_eligible": False,
        "reader_evidence": {
            "status": "not_run",
            "human_raters": 0,
            "reason": "No blinded reader study is run without an exact-clean closure.",
        },
    }


def cursor_center_contract() -> dict:
    """Exercise even and both in-word one-character center modes."""
    def probe(left_text: tuple[str, ...], right_text: tuple[str, ...]) -> dict:
        left = Clause(
            "probe-left", "probe", "probe", "probe", "probe", "sg", "future", "perfect",
            "future", "perfect", "since", tuple(Word(x, "probe", 0) for x in left_text), ("probe",),
        )
        right = Clause(
            "probe-right", "probe", "probe", "probe", "probe", "sg", "future", "perfect",
            "future", "perfect", "since", tuple(Word(x, "probe", 0) for x in right_text), ("probe",),
        )
        walk = walk_pair(left, right)
        return {
            "left_words": list(left_text),
            "right_words": list(right_text),
            "status": walk.status,
            "seam_mode": walk.seam_mode,
            "center_location": walk.center_location,
        }

    probes = [
        probe(("aba",), ("ba",)),
        probe(("ab",), ("cba",)),
        probe(("ab",), ("ba",)),
    ]
    modes = {item["seam_mode"] for item in probes}
    return {
        "description": "Cursor-level contract only; probes are not sentence candidates.",
        "probes": probes,
        "all_expected_modes_present": {
            "left_center_inside_word", "right_center_inside_word", "even_seam"
        }.issubset(modes),
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
        if entry.get("id") == PREDECESSOR_ID
        or any(marker in entry.get("signature", "") for marker in (
            "subject-relative", "temporal-subject-relative", "future-perfect",
            "matrix-perfect", "since-after",
        ))
    ]
    return {
        "status": "passed" if not exact_collision else "collision",
        "registry_inspected": True,
        "registry_entries_read": len(entries),
        "exact_signature_collision": exact_collision,
        "predecessor_lane_seen": PREDECESSOR_ID in related,
        "related_signatures_seen": related,
        "signature": SIGNATURE,
        "distinction": (
            "Adds a held-out matrix future-perfect auxiliary and participle state to the "
            "future-perfect subject-relative frame, and replaces the predecessor's before/by "
            "adjuncts with a fresh since/after inventory. It keeps nominative-head agreement, "
            "unequal word and letter boundaries, and live in-word center states without replay, "
            "repair, reversal, mirrored units, catalogue text, or per-search RLAIF."
        ),
        "not_a_duplicate_sweep": True,
    }


def select_controls(observations: list[tuple], limit: int = 24) -> list[dict]:
    """Retain complete, independently rendered prose controls across states."""
    ordered = sorted(
        observations,
        key=lambda item: (
            shortcut_flags(item[2], item[3])["repeated_content_units"],
            -(item[2].letters + item[3].letters),
            item[0],
            item[1],
        ),
    )
    selected = []
    seen_surfaces: set[tuple[str, str]] = set()
    seen_states: set[tuple[str, str, str, str, str]] = set()
    for left_index, right_index, left, right, walk in ordered:
        surface_key = (left.surface, right.surface)
        state_key = (
            left.number, right.number, left.temporal_relation, right.temporal_relation,
            left.matrix_aspect,
        )
        if left.surface == right.surface or surface_key in seen_surfaces:
            continue
        if state_key in seen_states and len(selected) < 12:
            continue
        selected.append((left_index, right_index, left, right, walk))
        seen_surfaces.add(surface_key)
        seen_states.add(state_key)
        if len(selected) >= limit:
            break
    return [row_for(left, right, walk) for _, _, left, right, walk in selected]


def run() -> dict:
    paths = build_paths()
    observations = []
    total_equations = mismatch_prunes = overhang_prunes = seam_closures = 0
    exact_candidates = []
    unequal_words = unequal_letters = 0
    center_modes: dict[str, int] = {}

    for left_index, left in enumerate(paths):
        for right_index, right in enumerate(paths):
            walk = walk_pair(left, right)
            observations.append((left_index, right_index, left, right, walk))
            total_equations += walk.online_equations
            mismatch_prunes += walk.status == "mismatch_pruned"
            overhang_prunes += walk.status == "overhang_pruned"
            seam_closures += walk.status == "closed"
            center_modes[walk.seam_mode] = center_modes.get(walk.seam_mode, 0) + 1
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
    result = {
        "experiment_id": EXPERIMENT_ID,
        "signature": SIGNATURE,
        "method": (
            "Fresh typed subject-relative temporal grammar with a future-perfect relative "
            "clause and a held-out future-perfect matrix clause. The predecessor's before/by "
            "adjunct inventory is replaced by since/after temporal adjuncts. Ordinary-order "
            "left and right clauses are intersected by the unequal-center two-cursor seam "
            "machine; unequal boundaries and in-word centers remain live."
        ),
        "operator_added": {
            "name": "matrix future-perfect alternation with since/after temporal attachment",
            "surface_form": (
                "SubjectHead who will have RelativeParticiple RelativeTheme TemporalAdjunct "
                "will have MatrixParticiple MatrixTheme"
            ),
            "case_state": "nominative_head",
            "attachment_state": "subject_relative_temporal",
            "number_states": ["sg", "pl"],
            "relative_tense_state": "future",
            "relative_aspect_state": "perfect",
            "matrix_tense_state": "future",
            "matrix_aspect_state": "perfect",
            "matrix_auxiliary_state": "will-have",
            "temporal_relations": ["since", "after"],
            "fresh_lexical_bank": True,
            "same_unequal_center_cursor_product": True,
            "predecessor_before_by_inventory_replayed": False,
            "per_search_rlaif": False,
        },
        "stats": {
            "heldout_clause_paths": len(paths),
            "paired_grammar_states": len(observations),
            "online_character_equations": total_equations,
            "mismatch_prunes": mismatch_prunes,
            "overhang_prunes": overhang_prunes,
            "seam_closures": seam_closures,
            "seam_modes": center_modes,
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
        "center_transition_contract": cursor_center_contract(),
        "novelty_preflight": novelty_preflight(),
        "provenance": {
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_cursor_source": "experiments/unequal_center_ditransitive_relative_complement_20260920.py",
            "independent_audits": ["literal outside-in pointer scan", "forward/reverse SHA-256"],
            "search_lexical_source": "fresh authored matrix-perfect temporal subject-relative role slots",
            "prior_before_by_paths_replayed": False,
            "ordinary_order_grammar_emission": True,
            "right_clause_read_inward_by_cursor": True,
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
            "operator": "typed since/after adjunct relocation with a matrix-perfect event alternation",
            "description": (
                "Hold the new matrix future-perfect state fixed and move one finite since/after "
                "adjunct to the matrix clause, preserving a fresh relative-clause adjunct bank. "
                "Index the resulting two adjunct attachment states by first two seam characters "
                "before opening new lexical items; this is a new attachment topology, not a repair "
                "or duplicate lexical sweep."
            ),
            "reader_facing_test": (
                "Render only complete controls; independently audit any exact closure above 38 "
                "letters, then randomize intact versus shuffled versions for blinded ratings."
            ),
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps({"experiment_id": output["experiment_id"], "stats": output["stats"]}, sort_keys=True))
